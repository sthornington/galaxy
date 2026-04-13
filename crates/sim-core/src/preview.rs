use std::mem::size_of;

use anyhow::{anyhow, bail};
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice, pod_read_unaligned};
use serde::{Deserialize, Serialize};

use crate::Vec3;

const PREVIEW_PACKET_MAGIC: [u8; 4] = *b"GPKT";
const PREVIEW_PACKET_VERSION: u32 = 1;

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
    pub galaxy_index: u32,
    pub component: u32,
    pub color_rgba: [f32; 4],
    pub intensity: f32,
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
    pub galaxy_index: u32,
    pub component: u32,
    pub color_rgba: [f32; 4],
    pub intensity: f32,
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

    out.clear();
    out.reserve(size_of::<PreviewPacketHeader>() + std::mem::size_of_val(particles));
    out.extend_from_slice(bytes_of(&header));
    out.extend_from_slice(cast_slice(particles));
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

    let particle_bytes = &bytes[header_len..];
    let expected_len = header.preview_count as usize * size_of::<PreviewPacketParticle>();
    if particle_bytes.len() != expected_len {
        return Err(anyhow!(
            "preview packet payload size mismatch: expected {} bytes for {} particles, got {}",
            expected_len,
            header.preview_count,
            particle_bytes.len()
        ));
    }

    let particle_stride = size_of::<PreviewPacketParticle>();
    let particles = particle_bytes
        .chunks_exact(particle_stride)
        .map(|chunk| pod_read_unaligned::<PreviewPacketParticle>(chunk))
        .map(|particle| PreviewParticle {
            position_kpc: particle.position_kpc,
            velocity_kms: particle.velocity_kms,
            mass_msun: particle.mass_msun,
            galaxy_index: particle.galaxy_index,
            component: particle.component,
            color_rgba: particle.color_rgba,
            intensity: particle.intensity,
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
