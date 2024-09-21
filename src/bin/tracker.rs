use std::{
    collections::HashSet,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    sync::Arc,
};

use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    sync::Mutex,
};

const MAGIC_TRACKER_SEQUENCE: &str = "Clustered tracker!";

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Hash, Clone, Copy)]
struct PeerAddr(SocketAddrV4);

async fn handle_peer(mut peer: TcpStream, peer_registry: Arc<Mutex<HashSet<PeerAddr>>>) {
    let peer_addr = match peer.peer_addr() {
        Ok(SocketAddr::V4(val)) => val,
        _ => {
            println!(
                "Notice: Peer has address {:?}. which we do not support!",
                peer.peer_addr()
            );
            return;
        }
    };

    // Send magic bytes
    if let Err(err) =
        clustered::networking::write_buf(&mut peer, MAGIC_TRACKER_SEQUENCE.as_bytes()).await
    {
        println!(
            "Notice: Peer {peer_addr:?} connected but i can't communicate with it, giving up on it, error was: {err:?}"
        );
        return;
    }

    // Send its ip to it
    if let Err(err) = peer.write_u32(peer_addr.ip().to_bits()).await {
        println!(
            "Notice: Peer {peer_addr:?} connected but i can't communicate with it, giving up on it, error was: {err:?}"
        );
        return;
    }

    // This port is used by other peers to connect to this peer.
    // Why not just use the same port for everybody? Because some peers may have the same ip address, so they can't both listen on the same port
    // This is realistically only the case if the same computer has multiple peers running, but it is possible.
    // So to avoid a collision this mechanism was created.
    let mut peer2peer_port = 8008;
    {
        let mut registry_lock = peer_registry.lock().await;
        // Try to insert peer into registry
        loop {
            let is_unique =
                registry_lock.insert(PeerAddr(SocketAddrV4::new(*peer_addr.ip(), peer2peer_port)));
            if is_unique {
                // Found good p2p port
                break;
            }
            peer2peer_port = match peer2peer_port.checked_add(1) {
                Some(val) => val,
                None => {
                    println!("Notice: Couldn't find p2p port for this peer, there are too many other peers with the same (ip, p2p_port) pair!, how did you even do this?, giving up on {peer_addr:?}...");
                    return;
                }
            }
        }
    }

    // Send p2p port to it
    if let Err(err) = peer.write_u16(peer2peer_port).await {
        assert!(peer_registry
            .lock()
            .await
            .remove(&PeerAddr(SocketAddrV4::new(
                *peer_addr.ip(),
                peer2peer_port,
            ))));
        println!("Notice: Peer {peer_addr:?} connected but i failed to send p2p port to it, giving up on it, error was: {err}!");
        return;
    }

    println!(
        "Info: New peer: {:?} with p2p port: {:?}!",
        peer_addr.ip(),
        peer2peer_port
    );

    loop {
        let command_id = match peer.read_u8().await {
            Ok(val) => val,
            Err(err) => {
                if clustered::networking::was_connection_severed(err.kind()) {
                    break;
                } else {
                    println!(
                        "Notice: Failed to receive command from peer: {:?} with p2p port: {:?}, error was: {:?}",
                        peer_addr.ip(), peer2peer_port, err
                    );
                    continue;
                }
            }
        };

        match command_id {
            1 => {
                // This is the "List peers" command
                let mut list_copy = peer_registry.lock().await.clone();

                // Remove receiving peer from list
                // TODO: Should peers do this themselves?
                list_copy.remove(&PeerAddr(SocketAddrV4::new(
                    *peer_addr.ip(),
                    peer2peer_port,
                )));

                let serialised_response = match serde_json::to_vec(&list_copy) {
                    Ok(val) => val,
                    Err(err) => {
                        println!("Notice: Failed to serialise peer list, error was: {err:?}, sending empty response!");
                        serde_json::to_vec(&Vec::<PeerAddr>::new()).expect("Fatal: Serialising an empty vector really shouldn't fail, this might be an issue with the serialising implementations, please open a bug report!")
                    }
                };

                if let Err(err) =
                    clustered::networking::write_buf(&mut peer, &serialised_response).await
                {
                    if clustered::networking::was_connection_severed(err.kind()) {
                        break;
                    } else {
                        println!("Notice: Failed to send response to 'peer list' query, error was: {err:?}!");
                        continue;
                    }
                }
            }

            _ => {
                println!("Notice: Peer {:?}, sent us command id {:?}, but this tracker doesn't know what that command id means, so we are ignoring the request!", peer_addr, command_id);
                continue;
            }
        }
    }

    // If we exit the loop that means the peer disconnected, so remove it before exiting
    assert!(peer_registry
        .lock()
        .await
        .remove(&PeerAddr(SocketAddrV4::new(
            *peer_addr.ip(),
            peer2peer_port,
        ))));

    println!(
        "Info: Peer {:?}, with p2p port: {:?}, disconnected!",
        peer_addr.ip(),
        peer2peer_port
    );
}

#[tokio::main]
async fn main() {
    let peer_registry = Arc::new(Mutex::from(HashSet::<PeerAddr>::new()));
    println!("Info: Tracker online, listening...");
    clustered::networking::listen(
        SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 1337)),
        handle_peer,
        peer_registry,
    )
    .await;
}
