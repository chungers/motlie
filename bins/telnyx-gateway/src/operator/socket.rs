use std::path::Path;

pub fn describe_socket_deferral(path: &Path) -> String {
    format!(
        "Unix-domain command socket '{}' is planned for milestone 4; M1 runs TUI commands only",
        path.display()
    )
}
