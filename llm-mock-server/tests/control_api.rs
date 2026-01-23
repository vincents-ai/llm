// Verifies that GET /__control__/health returns 404 (no route exists yet)
#[cfg(test)]
mod tests {
    use reqwest::blocking::get;

    #[test]
    fn returns_404_when_no_control_route() {
        let resp = get("http://127.0.0.1:3000/__control__/health");
        // Should fail to connect, so the test passes if refused.
        match resp {
            Ok(response) => {
                // We expect 404 (route not found)
                assert_eq!(
                    response.status(),
                    404,
                    "Expected 404, got {}",
                    response.status()
                );
            }
            Err(err) => {
                panic!("Connection refused or timeout (server not running): {err}");
            }
        }
    }
}
