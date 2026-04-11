use clap::Command;

pub fn root(name: &'static str) -> Command {
    Command::new(name)
}
