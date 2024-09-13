use std::{
    io::{self, ErrorKind},
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
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

type TaskQueueType = Arc<Mutex<Vec<Task>>>;

async fn return_data(data: Vec<u8>, return_addr: SocketAddrV4) {
    let mut connection = match TcpStream::connect(return_addr).await {
        Ok(val) => val,
        Err(err) => {
            println!(
                "Notice: Couldn't connect to peer {return_addr:?} to return data, error was: {err:?}!"
            );
            return;
        }
    };
    if let Err(err) = clustered::networking::write_buf(&mut connection, &data).await {
        println!("Notice: Couldn't send return data to peer {return_addr:?}, error was: {err:?}!");
    }
}

async fn consume_task(task: Task, device: &wgpu::Device, queue: &wgpu::Queue) {
    println!("Info: Consuming task!");
    let Some(result) = task.program.run(device, queue).await else {
        println!("Error: Failed to run task, discarding it!");
        return;
    };
    tokio::spawn(return_data(result, task.return_addr));
}

#[derive(Serialize, Deserialize, Debug)]
struct PeerAddr(SocketAddrV4);

async fn steal_task(
    task_queue: TaskQueueType,
    tracker_connection: Arc<Mutex<TcpStream>>,
) -> io::Result<()> {
    let peer_list = {
        let mut tracker_connection_lock = tracker_connection.lock().await;

        // Message id 1 is "get peer list" for tracker
        tracker_connection_lock.write_u8(1).await.map_err(|err| {
            io::Error::new(
                err.kind(),
                format!("Error: Couldn't send message id to tracker, error was: {err}!"),
            )
        })?;

        let raw_peer_list = clustered::networking::read_buf(&mut tracker_connection_lock)
            .await
            .map_err(|err| {
                io::Error::new(
                    err.kind(),
                    format!("Error: Couldn't receive peer list from tracker, error was: {err}!"),
                )
            })?;

        serde_json::from_slice::<Vec<PeerAddr>>(&raw_peer_list)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, format!("Error: Couldn't deserialise peer list received from tracker, error was: {err:?}!")))?
    };

    if peer_list.is_empty() {
        // Prevent a hot loop
        sleep(Duration::from_millis(100)).await;
    }

    for other_peer in peer_list {
        let mut other_peer_connection = match TcpStream::connect(other_peer.0).await {
            Ok(val) => val,
            Err(err) => {
                if err.kind() == ErrorKind::ConnectionRefused {
                    // Just assume that the client went offline in between us getting the peer list from the tracker and trying to connect to it
                    // So no print just ignore
                } else {
                    println!(
                        "Notice: Couldn't establish connection to peer: {:?}, while looking to steal tasks, error was: {err}!",
                        other_peer.0
                );
                }
                continue;
            }
        };

        let Ok(_) = clustered::networking::write_buf(
            &mut other_peer_connection,
            MAGIC_PEER2PEER_SEQUENCE.as_bytes(),
        )
        .await
        else {
            println!(
                "Notice: Couldn't send magic sequence to other peer: {:?}!",
                other_peer.0
            );
            continue;
        };

        // Message id 1 is "steal task" for peers
        let Ok(_) = other_peer_connection.write_u8(1).await else {
            println!(
                "Notice: Couldn't send message id to other peer: {:?} for task stealing!",
                other_peer.0
            );
            continue;
        };
        let Ok(raw_res) = clustered::networking::read_buf(&mut other_peer_connection).await else {
            println!("Notice: Couldn't receive task from other peer {other_peer:?}!");
            continue;
        };
        let Ok(res) = serde_json::from_slice::<'_, Option<Task>>(&raw_res) else {
            println!("Notice: Couldn't deserialise task from other peer {other_peer:?}!");
            continue;
        };
        if let Some(tsk) = res {
            let mut task_queue_lock = task_queue.lock().await;
            task_queue_lock.push(tsk);
            drop(task_queue_lock);
            println!("Info: Just stole a task from: {:?}!", other_peer);
            break;
        }
    }
    Ok(())
}

const MINIMUM_TASKS_TRESH: usize = 1;

async fn runner(task_queue: TaskQueueType, tracker_connection: Arc<Mutex<TcpStream>>) {
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

    async fn steal_task_wrapper(
        task_queue: TaskQueueType,
        tracker_connection: Arc<Mutex<TcpStream>>,
    ) {
        if let Err(err) = steal_task(task_queue, tracker_connection).await {
            if clustered::networking::was_connection_severed(err.kind()) {
                println!("FATAL: Lost connection to tracker!");
            } else {
                println!("{err}");
            }
        }
    }

    loop {
        let mut task_queue_guard = task_queue.lock().await;
        let mut task_queue_len = task_queue_guard.len();
        if let Some(tsk) = task_queue_guard.pop() {
            drop(task_queue_guard);
            task_queue_len -= 1;
            if task_queue_len <= MINIMUM_TASKS_TRESH {
                tokio::spawn(steal_task_wrapper(
                    task_queue.clone(),
                    tracker_connection.clone(),
                ));
            }
            consume_task(tsk, &device, &queue).await;
        } else {
            drop(task_queue_guard);
            // Queue is empty, there's no point in spawning task_queue to run concurrently as we need to wait for a task to be stolen anyways
            // This also ensures that steal_task doesn't get spammed in parallel when the queue is empty causing the equivalent of a fork bomb
            steal_task_wrapper(task_queue.clone(), tracker_connection.clone()).await;
        }
    }
}

async fn handle_other_peer(
    mut other_stream: TcpStream,
    task_queue: TaskQueueType,
) -> io::Result<()> {
    let magic_sequence = String::from_utf8(
        clustered::networking::read_buf(&mut other_stream).await?,
    )
    .map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Error: Bad magic sequence, it's not utf-8, error was: {err}"),
        )
    })?;

    if magic_sequence != MAGIC_PEER2PEER_SEQUENCE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Error: Magic sequence was valid utf-8, but it's not correct. Here it is: {magic_sequence:?}"),
        ));
    }

    loop {
        let message_id = other_stream.read_u8().await.map_err(|err| {
            io::Error::new(
                err.kind(),
                format!(
                    "Error: Failed to receive message id from peer {:?}, error was: {err}!",
                    other_stream.peer_addr()
                ),
            )
        })?;
        match message_id {
            1 => {
                // Other peer wants to steal from us
                // Let's just pick at random for now
                let mut task_queue_lock = task_queue.lock().await;
                let response = if task_queue_lock.len() <= 4 {
                    // We don't have enough tasks to benefit from giving to someone else
                    // by the time it takes to transfer the task and and receive the result we are better off just running the task ourselves
                    None
                } else {
                    task_queue_lock.pop()
                };
                drop(task_queue_lock);
                let serialised_response = serde_json::to_vec(&response)
                    .unwrap_or_else(|err| {
                        println!("Notice: Couldn't serialise task, sending empty response instead, this is probably a bug in the serialising implementation, error was: {err}!");
                        serde_json::to_vec(&Option::<Task>::None).unwrap()
                    });

                clustered::networking::write_buf(&mut other_stream, &serialised_response)
                    .await
                    .map_err(|err| {
                        io::Error::new(
                            err.kind(),
                            format!(
                                "Error: Failed to send task to peer: {:?}, error was: {err}",
                                other_stream.peer_addr()
                            ),
                        )
                    })?;
            }
            _ => {
                println!(
                    "Notice: Unknown message id({:?}) received from peer({:?})!",
                    message_id,
                    other_stream.peer_addr()
                )
            }
        }
    }
}

async fn start_peer(tracker_addr: SocketAddr) -> io::Result<(Ipv4Addr, TaskQueueType, TcpStream)> {
    let mut tracker_connection = TcpStream::connect(tracker_addr).await.map_err(|err| {
        io::Error::new(
            err.kind(),
            format!("Error: Connection to tracker failed, error was: {err}!"),
        )
    })?;

    let tracker_magic = clustered::networking::read_buf(&mut tracker_connection)
        .await
        .map_err(|err| {
            io::Error::new(
                err.kind(),
                format!("Error: Connection to tracker failed, error was: {err}!"),
            )
        })?;

    if tracker_magic != MAGIC_TRACKER_SEQUENCE.as_bytes() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Error: Bad magic: {:?} received from tacker: {:?}",
                String::from_utf8(tracker_magic),
                tracker_connection.peer_addr()
            ),
        ));
    }

    println!(
        "Info: Connected to tracker: {:?}!",
        tracker_connection.peer_addr()
    );

    let our_ip = Ipv4Addr::from_bits(tracker_connection.read_u32().await.map_err(|err| {
        io::Error::new(
            err.kind(),
            format!("Error: Connection to tracker failed, error was: {err}!"),
        )
    })?);

    let peer2peer_port = tracker_connection.read_u16().await.map_err(|err| {
        io::Error::new(
            err.kind(),
            format!("Error: Connection to tracker failed, error was: {err}!"),
        )
    })?;

    let queue: Arc<Mutex<Vec<Task>>> = Arc::from(Mutex::from(Vec::new()));

    {
        // Start listening for other peers

        async fn handle_other_peer_wrapper(other_stream: TcpStream, task_queue: TaskQueueType) {
            if let Err(err) = handle_other_peer(other_stream, task_queue).await {
                if !clustered::networking::was_connection_severed(err.kind()) {
                    println!("{err}");
                }
            }
        }

        tokio::spawn(clustered::networking::listen(
            SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, peer2peer_port)),
            handle_other_peer_wrapper,
            queue.clone(),
        ));
    }

    Ok((our_ip, queue, tracker_connection))
}

#[tokio::main]
async fn main() {
    let (our_ip, task_queue, tracker_connection) =
        start_peer(SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 1337)))
            .await
            .unwrap();

    tokio::spawn(runner(
        task_queue.clone(),
        Arc::new(Mutex::new(tracker_connection)),
    ));
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

    let test_program = serde_json::from_str::<SerialisableProgram>(&program_file_contents)
        .expect("Program file should be able to be deserialised!");
    println!("Program loaded!");
    let mut tq = Vec::new();
    for i in 0..10 {
        let tst_program = test_program.clone();
        let tsk_queue = task_queue.clone();
        tq.push(tokio::spawn(async move {
            let time_start = Instant::now();
            tsk_queue.lock().await.push(Task {
                program: tst_program,
                return_addr: SocketAddrV4::new(our_ip, 3499 + i),
            });
            let result_listener = TcpListener::bind(SocketAddrV4::new(our_ip, 3499 + i))
                .await
                .unwrap();
            let (mut result_stream, _) = result_listener.accept().await.unwrap();
            let raw_res = clustered::networking::read_buf(&mut result_stream)
                .await
                .unwrap();
            let time_end = Instant::now();
            assert!(raw_res.len() == core::mem::size_of::<f32>() * 4000 * 4000);
            println!("Took {}s", (time_end - time_start).as_secs_f32());
        }));
    }

    for f in tq {
        f.await.unwrap();
    }
    println!("Info: Exiting...");
}
