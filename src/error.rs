use thiserror::Error;

#[derive(Error, Debug)]
pub enum OcrError {
    #[error("Tensor error: {0}")]
    TensorError(String),

    #[error("Operator error in {op}: {msg}")]
    OperatorError { op: String, msg: String },

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Image error: {0}")]
    ImageError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("SafeTensors error: {0}")]
    SafeTensorsError(String),
}

pub type Result<T> = std::result::Result<T, OcrError>;
