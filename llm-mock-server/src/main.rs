use axum::Router;
use tokio::net::TcpListener;

#[tokio::main]
async fn main() {
    let app = Router::new(); // No routes yet
    let listener = TcpListener::bind("127.0.0.1:3000").await.unwrap();
    println!("Starting minimal Axum server at 127.0.0.1:3000");
    axum::serve(listener, app).await.unwrap();
}
