use std::{
    io,
    net::{Ipv4Addr, SocketAddrV4},
    sync::Arc,
    time::Duration,
};

use clustered::serialisable_program::SerialisableProgram;
use serde::{Deserialize, Serialize};
use tokio::{
    fs::OpenOptions,
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::Mutex,
    time::{sleep, Instant},
};
use wgpu::{DeviceDescriptor, InstanceDescriptor, RequestAdapterOptions};

const MAGIC_PEER2PEER_SEQUENCE: &str = "Clustered peer2peer, yay!";
const MAGIC_TRACKER_SEQUENCE: &str = "Clustered tracker!";

#[derive(Debug, Serialize, Deserialize)]
struct Task {
    return_addr: SocketAddrV4, // Where to return result
    program: SerialisableProgram,
}

type QueueType = Arc<Mutex<Vec<Task>>>;

async fn consume_task(task: Task, device: &wgpu::Device, queue: &wgpu::Queue) {
    println!("Info: Consuming task!");
    let result = task.program.run(device, queue).await;
    let mut connection = TcpStream::connect(task.return_addr).await.unwrap();
    clustered::networking::write_buf(&mut connection, &result)
        .await
        .unwrap();
}

#[derive(Serialize, Deserialize, Debug)]
struct PeerAddr(SocketAddrV4);

async fn steal_task(task_queue: QueueType, tracker_connection: Arc<Mutex<TcpStream>>) {
    let mut tracker_connection_lock = tracker_connection.lock().await;

    // Message id 1 is "get peer list" for tracker
    if let Err(err) = tracker_connection_lock.write_u8(1).await {
        match err.kind() {
            _ if clustered::networking::was_connection_severed(err.kind()) => {
                panic!("FATAL: Lost connection to tracker!")
            }

            _ => println!("Notice: Couldn't send message id to tracker: {:?}!", err),
        }

        return;
    }

    let peer_list = {
        let raw_peer_list =
            match clustered::networking::read_buf(&mut tracker_connection_lock).await {
                Ok(val) => val,
                Err(err) => {
                    println!(
                        "Notice: Couldn't recive peer list from tracker! Error was: {:?}",
                        err
                    );

                    return;
                }
            };

        match serde_json::from_slice::<Vec<PeerAddr>>(&raw_peer_list) {
            Ok(val) => val,
            Err(err) => {
                println!(
                    "Notice: Couldn't deserialise peer list received from tracker! Error was: {:?}",
                    err
                );
                return;
            }
        }
    };

    drop(tracker_connection_lock);

    if peer_list.is_empty() {
        // Prevent a hot loop
        sleep(Duration::from_millis(100)).await;
    }

    for other_peer in peer_list {
        let Ok(mut other_peer_connection) = TcpStream::connect(other_peer.0).await else {
            println!(
                "NOT-NORMAL: Couldn't establish connection to peer: {:?}, while looking to steal tasks!",
                other_peer.0
            );
            continue;
        };
        let Ok(_) = clustered::networking::write_buf(
            &mut other_peer_connection,
            MAGIC_PEER2PEER_SEQUENCE.as_bytes(),
        )
        .await
        else {
            println!(
                "NOT-NORMAL: Couldn't send magic sequence to other peer: {:?}!",
                other_peer.0
            );
            continue;
        };

        // Message id 1 is "steal task" for peers
        let Ok(_) = other_peer_connection.write_u8(1).await else {
            println!(
                "NOT-NORMAL: Couldn't send message id to other peer: {:?} for task stealing!",
                other_peer.0
            );
            continue;
        };
        let Ok(raw_res) = clustered::networking::read_buf(&mut other_peer_connection).await else {
            println!("NOT-NORMAL: Couldn't recive task from other peer!");
            continue;
        };
        let Ok(res) = serde_json::from_slice::<'_, Option<Task>>(&raw_res) else {
            println!("NOT-NORMAL: Couldn't deserialise task from other peer!");
            continue;
        };
        if let Some(tsk) = res {
            let mut task_queue_lock = task_queue.lock().await;
            task_queue_lock.push(tsk);
            drop(task_queue_lock);
            break;
        }
    }
}

const MINIMUM_TASKS_TRESH: usize = 1;

async fn runner(task_queue: QueueType, tracker_connection: Arc<Mutex<TcpStream>>) {
    let instance = wgpu::Instance::new(InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            compatible_surface: None,
            force_fallback_adapter: false,
            power_preference: wgpu::PowerPreference::None,
        })
        .await
        .expect("Should be able to acquire adapter!");
    println!("Runner is using {:?}", adapter.get_info());
    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::BUFFER_BINDING_ARRAY
                    | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )
        .await
        .expect("Should be able to get handle on device!");
    loop {
        let mut task_queue_guard = task_queue.lock().await;
        let mut task_queue_len = task_queue_guard.len();
        if let Some(tsk) = task_queue_guard.pop() {
            drop(task_queue_guard);
            task_queue_len -= 1;
            if task_queue_len <= MINIMUM_TASKS_TRESH {
                tokio::spawn(steal_task(task_queue.clone(), tracker_connection.clone()));
            }
            consume_task(tsk, &device, &queue).await;
        } else {
            drop(task_queue_guard);
            // Queue is empty, there's no point in spawning task_queue to run concurrently as we need to wait for a task to be stolen anyways
            // This also ensures that steal_task doesn't get spammed in parallel when the queue is empty causing the equivalent of a fork bomb
            steal_task(task_queue.clone(), tracker_connection.clone()).await;
        }
    }
}

async fn handle_other_peer(mut other_stream: TcpStream, task_queue: QueueType) -> io::Result<()> {
    let magic_sequence = String::from_utf8(
        clustered::networking::read_buf(&mut other_stream).await?,
    )
    .map_err(|err: std::string::FromUtf8Error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "NOT-NORMAL: Bad magic sequence, it's not utf-8! Error: {:?}",
                err
            ),
        )
    })?;

    if magic_sequence != MAGIC_PEER2PEER_SEQUENCE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "NOT-NORMAL: Magic sequence was valid utf-8, but it's not correct. Here it is: {:?}",
                magic_sequence
            ),
        ));
    }

    loop {
        let message_id = other_stream.read_u8().await?;
        match message_id {
            1 => {
                // Other peer wants to steal from us
                // Let's just pick at random for now
                let mut task_queue_lock = task_queue.lock().await;
                let response = task_queue_lock.pop();
                drop(task_queue_lock);
                let serialised_response = serde_json::to_vec(&response).unwrap();
                clustered::networking::write_buf(&mut other_stream, &serialised_response)
                    .await
                    .unwrap();
            }
            _ => {
                println!(
                    "NOT-NORMAL: Unknown message id({:?}) received from peer({:?})!",
                    message_id,
                    other_stream.peer_addr()
                )
            }
        }
    }
}

async fn listen_for_other_peers(task_queue: Arc<Mutex<Vec<Task>>>, peer2peer_port: u16) {
    let others_listener =
        TcpListener::bind(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, peer2peer_port))
            .await
            .unwrap();
    loop {
        match others_listener.accept().await {
            Ok((other_stream, _)) => {
                tokio::spawn(handle_other_peer(other_stream, task_queue.clone()));
            }
            Err(e) => println!(
                "NOT-NORMAL: A peer tried to connect to us but failed: {:?}",
                e
            ),
        }
    }
}

#[tokio::main]
async fn main() {
    let mut tracker_connection = TcpStream::connect(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 1337))
        .await
        .expect("Should be able to connect to tracker!");

    let tracker_magic = clustered::networking::read_buf(&mut tracker_connection)
        .await
        .expect("Should be able to communicate with tracker!");

    if tracker_magic != MAGIC_TRACKER_SEQUENCE.as_bytes() {
        panic!(
            "Bad magic: {:?} received from tacker: {:?}",
            String::from_utf8(tracker_magic),
            tracker_connection.peer_addr()
        );
    }

    println!(
        "Connected to tracker: {:?}!",
        tracker_connection.peer_addr()
    );

    let our_ip = Ipv4Addr::from_bits(
        tracker_connection
            .read_u32()
            .await
            .expect("Tracker should give us our ip!"),
    );

    let peer2peer_port = tracker_connection
        .read_u16()
        .await
        .expect("Tracker should give us a peer2peer port!");

    let queue: Arc<Mutex<Vec<Task>>> = Arc::from(Mutex::from(Vec::new()));
    tokio::spawn(runner(
        queue.clone(),
        Arc::new(Mutex::new(tracker_connection)),
    ));

    tokio::spawn(listen_for_other_peers(queue.clone(), peer2peer_port));
    sleep(Duration::MAX).await;

    let mut program_file = OpenOptions::new()
        .read(true)
        .open("program-capsule.json")
        .await
        .expect("Program file should exist!");
    let mut program_file_contents = String::new();
    program_file
        .read_to_string(&mut program_file_contents)
        .await
        .unwrap();
    drop(program_file);

    let test_program = serde_json::from_str(&program_file_contents)
        .expect("Program file should be able to be deserialised!");
    println!("Program loaded!");
    let time_start = Instant::now();
    queue.lock().await.push(Task {
        program: test_program,
        return_addr: SocketAddrV4::new(our_ip, 3499),
    });
    let result_listener = TcpListener::bind(SocketAddrV4::new(our_ip, 3499))
        .await
        .unwrap();
    let (mut result_stream, _) = result_listener.accept().await.unwrap();
    let raw_res = clustered::networking::read_buf(&mut result_stream)
        .await
        .unwrap();
    let time_end = Instant::now();
    assert!(raw_res.len() == core::mem::size_of::<f32>() * 4000 * 4000);
    println!("Took {}s", (time_end - time_start).as_secs_f32());
}
