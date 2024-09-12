use std::io::ErrorKind;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

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
