use std::mem::size_of;

use anyhow::{anyhow, bail};
use bytemuck::{Pod, Zeroable, bytes_of, pod_read_unaligned};
use serde::{Deserialize, Serialize};

use crate::Vec3;

const PREVIEW_PACKET_MAGIC: [u8; 4] = *b"GPKT";
const PREVIEW_PACKET_VERSION: u32 = 2;
const QUANT_MAX: f32 = 65535.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewFrame {
    pub sim_time_myr: f64,
    pub diagnostics: Diagnostics,
    pub particles: Vec<PreviewParticle>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewParticle {
    pub position_kpc: [f32; 3],
    pub velocity_kms: [f32; 3],
    pub mass_msun: f32,
    pub component: u32,
    /// Stellar age in 6.4 Myr ticks (255 = primordial / not a star).
    #[serde(default = "default_age_ticks")]
    pub age_ticks: u8,
}

fn default_age_ticks() -> u8 {
    255
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Diagnostics {
    pub particle_count: u64,
    pub preview_count: u32,
    pub sim_time_myr: f64,
    pub dt_myr: f64,
    pub kinetic_energy: f64,
    pub estimated_potential_energy: f64,
    pub total_momentum: Vec3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct PreviewPacketHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub particle_count: u64,
    pub preview_count: u32,
    pub reserved: u32,
    pub sim_time_myr: f64,
    pub dt_myr: f64,
    pub kinetic_energy: f64,
    pub estimated_potential_energy: f64,
    pub total_momentum: [f64; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct PreviewPacketParticle {
    pub position_kpc: [f32; 3],
    pub velocity_kms: [f32; 3],
    pub mass_msun: f32,
    pub component: u32,
}

/// Per-frame quantization ranges for the packed particle records; decode is
/// `min + quantized/65535 * scale` (mass is log2-encoded).
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct PreviewPacketQuantBlock {
    pub position_min_kpc: [f32; 3],
    pub position_scale_kpc: [f32; 3],
    pub velocity_min_kms: [f32; 3],
    pub velocity_scale_kms: [f32; 3],
    pub mass_log2_min: f32,
    pub mass_log2_scale: f32,
}

/// 16-byte wire record: positions and velocities quantized to u16 against the
/// frame's quant block, mass log-quantized to u16, component as a byte. This
/// halves the stream against raw f32 records, which matters at a quarter
/// million particles per frame: less bandwidth, less browser GC churn, and the
/// WebGL renderer dequantizes in the vertex shader for free.
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct PreviewWireParticle {
    pub position_q: [u16; 3],
    pub velocity_q: [u16; 3],
    pub mass_q: u16,
    pub component: u8,
    pub reserved: u8,
}

fn quantize(value: f32, min: f32, scale: f32) -> u16 {
    if scale <= 0.0 {
        return 0;
    }
    (((value - min) / scale * QUANT_MAX) as i32).clamp(0, QUANT_MAX as i32) as u16
}

fn dequantize(quantized: u16, min: f32, scale: f32) -> f32 {
    min + f32::from(quantized) / QUANT_MAX * scale
}

pub fn encode_preview_packet_into(
    out: &mut Vec<u8>,
    diagnostics: &Diagnostics,
    particles: &[PreviewPacketParticle],
) {
    let header = PreviewPacketHeader {
        magic: PREVIEW_PACKET_MAGIC,
        version: PREVIEW_PACKET_VERSION,
        particle_count: diagnostics.particle_count,
        preview_count: particles.len() as u32,
        reserved: 0,
        sim_time_myr: diagnostics.sim_time_myr,
        dt_myr: diagnostics.dt_myr,
        kinetic_energy: diagnostics.kinetic_energy,
        estimated_potential_energy: diagnostics.estimated_potential_energy,
        total_momentum: [
            diagnostics.total_momentum.x,
            diagnostics.total_momentum.y,
            diagnostics.total_momentum.z,
        ],
    };

    let mut position_min = [f32::INFINITY; 3];
    let mut position_max = [f32::NEG_INFINITY; 3];
    let mut velocity_min = [f32::INFINITY; 3];
    let mut velocity_max = [f32::NEG_INFINITY; 3];
    let mut mass_log_min = f32::INFINITY;
    let mut mass_log_max = f32::NEG_INFINITY;
    for particle in particles {
        for axis in 0..3 {
            position_min[axis] = position_min[axis].min(particle.position_kpc[axis]);
            position_max[axis] = position_max[axis].max(particle.position_kpc[axis]);
            velocity_min[axis] = velocity_min[axis].min(particle.velocity_kms[axis]);
            velocity_max[axis] = velocity_max[axis].max(particle.velocity_kms[axis]);
        }
        let mass_log = particle.mass_msun.max(1.0).log2();
        mass_log_min = mass_log_min.min(mass_log);
        mass_log_max = mass_log_max.max(mass_log);
    }
    if particles.is_empty() {
        position_min = [0.0; 3];
        position_max = [0.0; 3];
        velocity_min = [0.0; 3];
        velocity_max = [0.0; 3];
        mass_log_min = 0.0;
        mass_log_max = 0.0;
    }
    let quant = PreviewPacketQuantBlock {
        position_min_kpc: position_min,
        position_scale_kpc: [
            position_max[0] - position_min[0],
            position_max[1] - position_min[1],
            position_max[2] - position_min[2],
        ],
        velocity_min_kms: velocity_min,
        velocity_scale_kms: [
            velocity_max[0] - velocity_min[0],
            velocity_max[1] - velocity_min[1],
            velocity_max[2] - velocity_min[2],
        ],
        mass_log2_min: mass_log_min,
        mass_log2_scale: mass_log_max - mass_log_min,
    };

    out.clear();
    out.reserve(
        size_of::<PreviewPacketHeader>()
            + size_of::<PreviewPacketQuantBlock>()
            + particles.len() * size_of::<PreviewWireParticle>(),
    );
    out.extend_from_slice(bytes_of(&header));
    out.extend_from_slice(bytes_of(&quant));
    for particle in particles {
        let wire = PreviewWireParticle {
            position_q: [
                quantize(particle.position_kpc[0], quant.position_min_kpc[0], quant.position_scale_kpc[0]),
                quantize(particle.position_kpc[1], quant.position_min_kpc[1], quant.position_scale_kpc[1]),
                quantize(particle.position_kpc[2], quant.position_min_kpc[2], quant.position_scale_kpc[2]),
            ],
            velocity_q: [
                quantize(particle.velocity_kms[0], quant.velocity_min_kms[0], quant.velocity_scale_kms[0]),
                quantize(particle.velocity_kms[1], quant.velocity_min_kms[1], quant.velocity_scale_kms[1]),
                quantize(particle.velocity_kms[2], quant.velocity_min_kms[2], quant.velocity_scale_kms[2]),
            ],
            mass_q: quantize(particle.mass_msun.max(1.0).log2(), quant.mass_log2_min, quant.mass_log2_scale),
            // Low byte: component id. High byte (from the backend's packed
            // component word): stellar age in 6.4 Myr ticks, 255 = old.
            component: (particle.component & 0xFF) as u8,
            reserved: ((particle.component >> 8) & 0xFF) as u8,
        };
        out.extend_from_slice(bytes_of(&wire));
    }
}

pub fn decode_preview_packet(bytes: &[u8]) -> anyhow::Result<PreviewFrame> {
    let header_len = size_of::<PreviewPacketHeader>();
    if bytes.len() < header_len {
        bail!(
            "preview packet too short: {} < header size {}",
            bytes.len(),
            header_len
        );
    }

    let header = pod_read_unaligned::<PreviewPacketHeader>(&bytes[..header_len]);
    if header.magic != PREVIEW_PACKET_MAGIC {
        bail!("invalid preview packet magic");
    }
    if header.version != PREVIEW_PACKET_VERSION {
        bail!(
            "unsupported preview packet version {}",
            header.version
        );
    }

    let quant_len = size_of::<PreviewPacketQuantBlock>();
    if bytes.len() < header_len + quant_len {
        bail!("preview packet too short for quant block");
    }
    let quant =
        pod_read_unaligned::<PreviewPacketQuantBlock>(&bytes[header_len..header_len + quant_len]);

    let particle_bytes = &bytes[header_len + quant_len..];
    // Compare in u64: a hostile preview_count times the stride can overflow usize
    // on wasm32.
    let expected_len = header.preview_count as u64 * size_of::<PreviewWireParticle>() as u64;
    if particle_bytes.len() as u64 != expected_len {
        return Err(anyhow!(
            "preview packet payload size mismatch: expected {} bytes for {} particles, got {}",
            expected_len,
            header.preview_count,
            particle_bytes.len()
        ));
    }

    let particle_stride = size_of::<PreviewWireParticle>();
    let particles = particle_bytes
        .chunks_exact(particle_stride)
        .map(|chunk| pod_read_unaligned::<PreviewWireParticle>(chunk))
        .map(|wire| PreviewParticle {
            position_kpc: [
                dequantize(wire.position_q[0], quant.position_min_kpc[0], quant.position_scale_kpc[0]),
                dequantize(wire.position_q[1], quant.position_min_kpc[1], quant.position_scale_kpc[1]),
                dequantize(wire.position_q[2], quant.position_min_kpc[2], quant.position_scale_kpc[2]),
            ],
            velocity_kms: [
                dequantize(wire.velocity_q[0], quant.velocity_min_kms[0], quant.velocity_scale_kms[0]),
                dequantize(wire.velocity_q[1], quant.velocity_min_kms[1], quant.velocity_scale_kms[1]),
                dequantize(wire.velocity_q[2], quant.velocity_min_kms[2], quant.velocity_scale_kms[2]),
            ],
            mass_msun: dequantize(wire.mass_q, quant.mass_log2_min, quant.mass_log2_scale).exp2(),
            component: u32::from(wire.component),
            age_ticks: wire.reserved,
        })
        .collect();

    Ok(PreviewFrame {
        sim_time_myr: header.sim_time_myr,
        diagnostics: Diagnostics {
            particle_count: header.particle_count,
            preview_count: header.preview_count,
            sim_time_myr: header.sim_time_myr,
            dt_myr: header.dt_myr,
            kinetic_energy: header.kinetic_energy,
            estimated_potential_energy: header.estimated_potential_energy,
            total_momentum: Vec3::new(
                header.total_momentum[0],
                header.total_momentum[1],
                header.total_momentum[2],
            ),
        },
        particles,
    })
}
