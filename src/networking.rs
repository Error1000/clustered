use std::{future::Future, io::ErrorKind, net::SocketAddr};

use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

pub async fn read_buf(connection: &mut tokio::net::TcpStream) -> std::io::Result<Vec<u8>> {
    let nbytes = connection.read_u64().await?;
    let mut buf = vec![0u8; nbytes.try_into().unwrap()];
    connection.read_exact(&mut buf).await?;
    Ok(buf)
}

pub async fn write_buf(connection: &mut tokio::net::TcpStream, buf: &[u8]) -> std::io::Result<()> {
    connection.write_u64(buf.len().try_into().unwrap()).await?;
    connection.write_all(buf).await?;
    Ok(())
}

pub async fn listen<F, Fut, ExtraData>(listen_addr: SocketAddr, handler: F, extra: ExtraData)
where
    F: Fn(TcpStream, ExtraData) -> Fut,
    ExtraData: Clone,
    Fut: Future<Output = ()> + Send + 'static,
{
    let listener = match TcpListener::bind(listen_addr).await {
        Ok(val) => val,
        Err(err) => {
            println!(
                "Error: Unable to bind to address {:?} for listening, error was: {:?}!",
                listen_addr, err
            );
            return;
        }
    };

    loop {
        match listener.accept().await {
            Ok((connection, _)) => {
                tokio::spawn(handler(connection, extra.clone()));
            }
            Err(err) => {
                println!("Notice: Unable to accept a connection, error was: {err:?}!");
            }
        }
    }
}

pub fn was_connection_severed(err_kind: ErrorKind) -> bool {
    matches!(
        err_kind,
        ErrorKind::NotConnected
            | ErrorKind::BrokenPipe
            | ErrorKind::ConnectionAborted
            | ErrorKind::ConnectionReset
            | ErrorKind::UnexpectedEof
    )
}
