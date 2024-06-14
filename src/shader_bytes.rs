pub trait ShaderBytesSize {
    // NOTE: By *not* taking a self we explicitly disallow dynamically sized types and unsized types
    // Because working with consistently sized types is overall better (opinion)
    // even if those types then need to contain pointers to dynamically sized data
    fn shader_bytes_size() -> usize;
}

pub trait IntoShaderBytes: ShaderBytesSize {
    fn to_shader_bytes(&self, res: &mut [u8]);
}

pub trait FromShaderBytes: ShaderBytesSize {
    fn from_shader_bytes(buf: &[u8]) -> Self;
}

impl ShaderBytesSize for u32 {
    fn shader_bytes_size() -> usize {
        core::mem::size_of::<Self>()
    }
}

impl IntoShaderBytes for u32 {
    fn to_shader_bytes(&self, res: &mut [u8]) {
        for (i, e) in self.to_le_bytes().iter().enumerate() {
            res[i] = *e;
        }
    }
}

impl FromShaderBytes for u32 {
    fn from_shader_bytes(buf: &[u8]) -> Self {
        Self::from_le_bytes(buf.try_into().unwrap())
    }
}

impl ShaderBytesSize for f32 {
    fn shader_bytes_size() -> usize {
        core::mem::size_of::<Self>()
    }
}

impl IntoShaderBytes for f32 {
    fn to_shader_bytes(&self, res: &mut [u8]) {
        for (i, e) in self.to_le_bytes().iter().enumerate() {
            res[i] = *e;
        }
    }
}

impl FromShaderBytes for f32 {
    fn from_shader_bytes(buf: &[u8]) -> Self {
        Self::from_le_bytes(buf.try_into().unwrap())
    }
}

impl ShaderBytesSize for f64 {
    fn shader_bytes_size() -> usize {
        core::mem::size_of::<Self>()
    }
}

impl IntoShaderBytes for f64 {
    fn to_shader_bytes(&self, res: &mut [u8]) {
        for (i, e) in self.to_le_bytes().iter().enumerate() {
            res[i] = *e;
        }
    }
}

impl FromShaderBytes for f64 {
    fn from_shader_bytes(buf: &[u8]) -> Self {
        Self::from_le_bytes(buf.try_into().unwrap())
    }
}
