use std::{
    collections::HashMap,
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
    net::TcpStream,
    sync::{Mutex, RwLock, Semaphore},
    time::{sleep, Instant},
};
use uuid::Uuid;
use wgpu::{DeviceDescriptor, InstanceDescriptor, RequestAdapterOptions};

const MAGIC_PEER2PEER_SEQUENCE: &str = "Clustered peer2peer, yay!";
const MAGIC_TRACKER_SEQUENCE: &str = "Clustered tracker!";

const MINIMUM_TASKS_BEFORE_START_STEALING_TRESH: usize = 5; // We won't steal if we have more than this number of tasks
const NO_STEAL_TRESHOLD: usize = 1; // No stealing will be allowed if we have less than this number of tasks

#[derive(Debug, Serialize, Deserialize)]
struct Task {
    return_addr: SocketAddrV4, // Where to return result
    program: SerialisableProgram,
    id: u128,
}

type TaskQueueType = Arc<Mutex<Vec<Task>>>;
type BufferRegistryType = Arc<RwLock<HashMap<Uuid, Vec<u8>>>>;
type NotifierRegistryType = Arc<RwLock<HashMap<Uuid, Arc<Semaphore>>>>;

async fn connect_to_other_peer(other_peer_addr: SocketAddr) -> io::Result<TcpStream> {
    let mut other_peer_connection = TcpStream::connect(other_peer_addr).await.map_err(|err| {
        io::Error::new(
            err.kind(),
            format!("{err}\nWhile connecting to other peer: {other_peer_addr}"),
        )
    })?;

    clustered::networking::write_buf(
        &mut other_peer_connection,
        MAGIC_PEER2PEER_SEQUENCE.as_bytes(),
    )
    .await
    .map_err(|err| {
        io::Error::new(
            err.kind(),
            format!("{err}\nWhile sending magic sequence to other peer: {other_peer_addr}"),
        )
    })?;

    Ok(other_peer_connection)
}

async fn connect_to_tracker(tracker_addr: SocketAddr) -> io::Result<(Ipv4Addr, u16, TcpStream)> {
    let mut tracker_connection = TcpStream::connect(tracker_addr).await.map_err(|err| {
        io::Error::new(
            err.kind(),
            format!("{err}\nWhile connecting to tracker: {tracker_addr}"),
        )
    })?;

    let tracker_magic = clustered::networking::read_buf(&mut tracker_connection)
        .await
        .map_err(|err| {
            io::Error::new(
                err.kind(),
                format!("{err}\nWhile receiving magic sequence from tracker: {tracker_addr}"),
            )
        })?;

    if tracker_magic != MAGIC_TRACKER_SEQUENCE.as_bytes() {
        return Err(io::Error::new(
            ErrorKind::Other,
            format!(
                "Bad magic {:?} received from tracker: {tracker_addr}!",
                String::from_utf8(tracker_magic)
            ),
        ));
    }

    let our_ip = Ipv4Addr::from_bits(tracker_connection.read_u32().await.map_err(|err| {
        io::Error::new(
            err.kind(),
            format!("{err}\nWhile receiving ip address from tracker: {tracker_addr}"),
        )
    })?);

    let peer2peer_port = tracker_connection.read_u16().await.map_err(|err| {
        io::Error::new(
            err.kind(),
            format!("{err}\nWhile receiving p2p port from tracker: {tracker_addr}"),
        )
    })?;

    Ok((our_ip, peer2peer_port, tracker_connection))
}

async fn return_data(
    data: Vec<u8>,
    return_addr: SocketAddrV4,
    task_id: Uuid,
    output_buffer_registry: BufferRegistryType,
    notifier_registry: NotifierRegistryType,
) {
    // We could test if the return_addr is ourselves, but it's easier to just search for the uuid in our registry
    // and if we have it then the return_addr is ourselves otherwise it's someone else and we need to connect to them.
    let mut buf_registry_write_lock = output_buffer_registry.write().await;
    if let Some(local_buf) = buf_registry_write_lock.get_mut(&task_id) {
        *local_buf = data;
        drop(buf_registry_write_lock);
        if let Some(notifier) = notifier_registry.read().await.get(&task_id) {
            notifier.add_permits(Semaphore::MAX_PERMITS);
        }
    } else {
        drop(buf_registry_write_lock);
        let mut other_peer_connection =
            match connect_to_other_peer(SocketAddr::V4(return_addr)).await {
                Ok(val) => val,
                Err(err) => {
                    if !clustered::networking::was_connection_severed(err.kind()) {
                        println!("Error:");
                        println!("{err}");
                        println!("While returning data to other peer: {return_addr}");
                    }
                    return;
                }
            };

        // Message id 2 is "return result" for peers
        if let Err(err) = other_peer_connection.write_u8(2).await {
            println!("Error: {err}");
            println!("While sending message id to other peer: {return_addr}");
            println!("While returning data to other peer: {return_addr}");
            return;
        };

        if let Err(err) = other_peer_connection.write_u128(task_id.as_u128()).await {
            println!("Error: {err}");
            println!("While sending task uuid to other peer: {return_addr}");
            println!("While returning data to other peer: {return_addr}");
            return;
        }

        if let Err(err) = clustered::networking::write_buf(&mut other_peer_connection, &data).await
        {
            println!("Error: {err}");
            println!("While sending return data to other peer: {return_addr}");
            println!("While returning data to other peer: {return_addr}");
        }
    }
}

async fn consume_task(
    task: Task,
    output_buffer_registry: BufferRegistryType,
    notifier_registry: NotifierRegistryType,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    println!("Info: Consuming task!");
    let task_uuid = Uuid::from_u128(task.id);
    let Some(result) = task.program.run(device, queue).await else {
        println!("Error: Failed to run task, discarding it!");
        return;
    };
    tokio::spawn(return_data(
        result,
        task.return_addr,
        task_uuid,
        output_buffer_registry,
        notifier_registry,
    ));
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
                format!(
                    "{err}\nWhile sending message id to tracker\nWhile attempting to steal tasks"
                ),
            )
        })?;

        let raw_peer_list = clustered::networking::read_buf(&mut tracker_connection_lock)
            .await
            .map_err(|err| {
                io::Error::new(
                    err.kind(),
                    format!("{err}\nWhile receiving peer list from tracker\nWhile attempting to steal tasks"),
                )
            })?;

        serde_json::from_slice::<Vec<PeerAddr>>(&raw_peer_list)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, format!("{err}\nWhile deserialising peer list received from tracker\nWhile attempting to steal tasks")))?
    };

    if peer_list.is_empty() {
        // Prevent a hot loop
        sleep(Duration::from_millis(100)).await;
    }

    for other_peer in peer_list {
        let mut other_peer_connection =
            match connect_to_other_peer(SocketAddr::V4(other_peer.0)).await {
                Ok(val) => val,
                Err(err) => {
                    // Connection refused might happen if the peer disconnects after we have gotten the peer list from the tracker
                    // but before we try to connect
                    if !clustered::networking::was_connection_severed(err.kind())
                        && err.kind() != ErrorKind::ConnectionRefused
                    {
                        println!("Notice:");
                        println!("{err}");
                        println!(
                            "While attempting to steal task from other peer: {:?}",
                            other_peer.0
                        );
                    }
                    continue;
                }
            };

        // Message id 1 is "steal task" for peers
        if let Err(err) = other_peer_connection.write_u8(1).await {
            if !clustered::networking::was_connection_severed(err.kind()) {
                println!("Notice:");
                println!("{err}");
                println!("While sending message id to other peer: {:?}", other_peer.0);
                println!(
                    "While attempting to steal task from other peer: {:?}",
                    other_peer.0
                );
            }
            continue;
        };

        let raw_res = match clustered::networking::read_buf(&mut other_peer_connection).await {
            Ok(val) => val,
            Err(err) => {
                if !clustered::networking::was_connection_severed(err.kind()) {
                    println!("Notice:");
                    println!("{err}");
                    println!("While receiveing task from other peer: {:?}", other_peer.0);
                    println!(
                        "While attempting to steal task from other peer: {:?}",
                        other_peer.0
                    );
                }
                continue;
            }
        };

        drop(other_peer_connection);

        let res: Option<Task> = match serde_json::from_slice(&raw_res) {
            Ok(val) => val,
            Err(err) => {
                println!("Notice:");
                println!("{err}");
                println!("While deserialising task received from other peer {other_peer:?}!");
                println!(
                    "While attempting to steal task from other peer: {:?}",
                    other_peer.0
                );
                continue;
            }
        };

        if let Some(tsk) = res {
            println!("Info: Just stole a task, from: {:?}!", other_peer.0);
            task_queue.lock().await.push(tsk);
            break;
        }
    }
    Ok(())
}

async fn runner(
    task_queue: TaskQueueType,
    output_buffer_registry: BufferRegistryType,
    notifier_registry: NotifierRegistryType,
    tracker_connection: Arc<Mutex<TcpStream>>,
) {
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
                println!("Error:");
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
            if task_queue_len <= MINIMUM_TASKS_BEFORE_START_STEALING_TRESH {
                tokio::spawn(steal_task_wrapper(
                    task_queue.clone(),
                    tracker_connection.clone(),
                ));
            }
            consume_task(
                tsk,
                output_buffer_registry.clone(),
                notifier_registry.clone(),
                &device,
                &queue,
            )
            .await;
        } else {
            drop(task_queue_guard);
            // Queue is empty, there's no point in spawning steal_task to run concurrently as we need to wait for a task to be stolen anyways
            // This also ensures that steal_task doesn't get spammed in parallel when the queue is empty causing the equivalent of a fork bomb
            steal_task_wrapper(task_queue.clone(), tracker_connection.clone()).await;
        }
    }
}

async fn handle_other_peer(
    mut other_stream: TcpStream,
    task_queue: TaskQueueType,
    output_buffer_registry: BufferRegistryType,
    notifier_registry: NotifierRegistryType,
) -> io::Result<()> {
    let magic_sequence = String::from_utf8(
        clustered::networking::read_buf(&mut other_stream).await?,
    )
    .map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Error: {err}\nWhile parsing magic sequence"),
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
                    "Error: {err}\nWhile receiving message id from peer {:?}",
                    other_stream.peer_addr()
                ),
            )
        })?;
        match message_id {
            1 => {
                // Other peer wants to steal from us
                // TODO: We just pick at random for now
                let mut task_queue_lock = task_queue.lock().await;
                let response = if task_queue_lock.len() <= NO_STEAL_TRESHOLD {
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
                                "Error: {err}\n While sending task to peer: {:?}",
                                other_stream.peer_addr()
                            ),
                        )
                    })?;
            }
            2 => {
                // Other peer wants to send us a task result
                let task_uuid = Uuid::from_u128(
                    other_stream.read_u128().await.map_err(|err| {
                    io::Error::new(
                        err.kind(),
                        format!(
                            "Error: {err}\nWhile receiveing uuid from peer {:?}\nWhile handling return task result message from peer {:?}",
                            other_stream.peer_addr(), other_stream.peer_addr()
                        ),
                    )
                })?
                );

                let data = clustered::networking::read_buf(&mut other_stream).await.map_err(|err| {
                    io::Error::new(
                        err.kind(),
                        format!(
                            "Error: {err}\n While receiveing buffer data from peer {:?}\nWhile handling return task result message from peer {:?}",
                            other_stream.peer_addr(), other_stream.peer_addr()
                        ),
                    )
                })?;

                if let Some(buf) = output_buffer_registry.write().await.get_mut(&task_uuid) {
                    *buf = data;
                } else {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        format!("Error: Task UUID {task_uuid}, received from peer not found in our buffer registry!"),
                    ));
                };

                if let Some(notifier) = notifier_registry.read().await.get(&task_uuid) {
                    notifier.add_permits(Semaphore::MAX_PERMITS);
                }
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

#[tokio::main]
async fn main() {
    let (our_ip, peer2peer_port, tracker_connection) =
        connect_to_tracker(SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 1337)))
            .await
            .unwrap_or_else(|err| panic!("FATAL:\n{err}"));

    println!(
        "Info: Connected to tracker: {:?}!",
        tracker_connection.peer_addr()
    );

    let task_queue: TaskQueueType = Default::default();
    let output_buffer_registry: BufferRegistryType = Default::default();
    let notifier_registry: NotifierRegistryType = Default::default();

    {
        // Start listening for other peers

        async fn handle_other_peer_wrapper(
            other_stream: TcpStream,
            extra: (TaskQueueType, BufferRegistryType, NotifierRegistryType),
        ) {
            if let Err(err) = handle_other_peer(other_stream, extra.0, extra.1, extra.2).await {
                if !clustered::networking::was_connection_severed(err.kind()) {
                    println!("{err}");
                }
            }
        }

        tokio::spawn(clustered::networking::listen(
            SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, peer2peer_port)),
            handle_other_peer_wrapper,
            (
                task_queue.clone(),
                output_buffer_registry.clone(),
                notifier_registry.clone(),
            ),
        ));
    }

    tokio::spawn(runner(
        task_queue.clone(),
        output_buffer_registry.clone(),
        notifier_registry.clone(),
        Arc::new(Mutex::new(tracker_connection)),
    ));

    // And now do normal peer stuff, like adding tasks to the queue and waiting for the results
    // sleep(Duration::MAX).await;

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
    for _ in 0..30 {
        let time_start = Instant::now();
        let task_id = Uuid::now_v7();
        output_buffer_registry
            .write()
            .await
            .insert(task_id, Vec::new());
        notifier_registry
            .write()
            .await
            .insert(task_id, Arc::from(Semaphore::new(0)));
        task_queue.lock().await.push(Task {
            program: test_program.clone(),
            return_addr: SocketAddrV4::new(our_ip, peer2peer_port),
            id: task_id.as_u128(),
        });

        let buf_reg_clone = output_buffer_registry.clone();
        let notif_reg_clone = notifier_registry.clone();
        tq.push(tokio::spawn(async move {
            let sem = notif_reg_clone
                .read()
                .await
                .get(&task_id)
                .expect("Task should have notifier!")
                .clone();

            let _ = sem.acquire().await.expect("Semaphore shouldn't close!");
            let buf_reg_lock = buf_reg_clone.read().await;
            let raw_res = buf_reg_lock
                .get(&task_id)
                .expect("Task should have output buffer!");
            assert!(raw_res.len() == core::mem::size_of::<f32>() * 4000 * 4000);
            let time_end = Instant::now();
            drop(buf_reg_lock);

            buf_reg_clone.write().await.remove(&task_id);
            notif_reg_clone.write().await.remove(&task_id);
            println!("Took: {}s!", (time_end - time_start).as_secs_f32());
        }));
    }

    for f in tq {
        f.await.unwrap();
    }

    while !task_queue.lock().await.is_empty() {
        sleep(Duration::from_millis(10)).await;
        tokio::task::yield_now().await;
    }

    assert!(output_buffer_registry.read().await.is_empty());
    assert!(notifier_registry.read().await.is_empty());
    assert!(task_queue.lock().await.is_empty());

    println!("Info(HACK: because i can't properly wait for all tasks to finish correctly yet): Press any key to exit...");
    {
        let mut junk_buf = Vec::new();
        let _ = tokio::io::stdin().read(&mut junk_buf).await.unwrap();
    }
}
