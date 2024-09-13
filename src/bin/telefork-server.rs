use std::net::{Ipv4Addr, SocketAddrV4};

use clustered::serialisable_program::SerialisableProgram;

use tokio::{net::TcpListener, time::Instant};
use wgpu::{DeviceDescriptor, InstanceDescriptor, RequestAdapterOptions};

#[tokio::main]
async fn main() {
    let instance = wgpu::Instance::new(InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            compatible_surface: None,
            force_fallback_adapter: false,
            power_preference: wgpu::PowerPreference::HighPerformance,
        })
        .await
        .unwrap();
    println!("Using {:?}", adapter.get_info());
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
        .unwrap();

    println!("Listening...");
    let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 1337))
        .await
        .unwrap();
    loop {
        let (mut connection, _) = listener.accept().await.unwrap();
        println!("Connection from {:?} accepted!", connection.peer_addr());
        let program_capsule: SerialisableProgram = serde_json::from_slice(
            &clustered::networking::read_buf(&mut connection)
                .await
                .unwrap(),
        )
        .unwrap();
        println!("Received and deserialised program!");
        let time_before = Instant::now();
        let res = program_capsule.run(&device, &queue).await.unwrap();
        let time_after = Instant::now();
        println!("Took: {:?}s!", (time_after - time_before).as_secs_f32());
        println!("Sending result...");
        clustered::networking::write_buf(&mut connection, &res)
            .await
            .unwrap();
    }
}
