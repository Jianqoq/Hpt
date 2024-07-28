use std::sync::Arc;

pub struct ArcString {
    inner: Arc<String>,
}

impl ArcString {
    pub fn new(s: &str) -> Self {
        Self {
            inner: Arc::new(s.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        &self.inner
    }
}

impl Into<ArcString> for String {
    fn into(self) -> ArcString {
        ArcString {
            inner: Arc::new(self),
        }
    }
}

impl Into<ArcString> for &str {
    fn into(self) -> ArcString {
        ArcString {
            inner: Arc::new(self.to_string()),
        }
    }
}

impl Into<ArcString> for Arc<String> {
    fn into(self) -> ArcString {
        ArcString {
            inner: self,
        }
    }
}
