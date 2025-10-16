use polars::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};

// ==============================================================================
// SECTION 1: NORMALIZATION AND FEATURE ENGINEERING HELPERS
// ==============================================================================

/// Engineers a feature by categorizing CAN IDs based on their frequency.
fn normalize_can_id_by_frequency(df: &mut DataFrame, column_name: &str) -> Result<()> {
    // Calculate the frequency of each unique CAN ID
    let id_counts = df
        .column(column_name)?
        .value_counts(true, false)?;
    
    let ids = id_counts.column("CAN_ID")?.u32()?;
    let counts = id_counts.column("counts")?.u32()?;
    
    // Create mapping from CAN_ID to category
    let mut id_to_category: HashMap<u32, u8> = HashMap::new();
    
    for (id_opt, count_opt) in ids.into_iter().zip(counts.into_iter()) {
        if let (Some(id), Some(count)) = (id_opt, count_opt) {
            let category = assign_category(count);
            id_to_category.insert(id, category);
        }
    }
    
    // Map CAN_IDs to categories
    let can_ids = df.column(column_name)?.u32()?;
    let categories: Vec<Option<u8>> = can_ids
        .into_iter()
        .map(|id_opt| id_opt.and_then(|id| id_to_category.get(&id).copied()))
        .collect();
    
    let normalized_column_name = format!("{}_Norm", column_name);
    let category_series = Series::new(&normalized_column_name, categories);
    df.with_column(category_series)?;
    
    Ok(())
}

/// Assign category based on count with non-overlapping bins
fn assign_category(count: u32) -> u8 {
    if count >= 12000 {
        1
    } else if count >= 5000 {
        2
    } else if count >= 2500 {
        3
    } else {
        4
    }
}

/// Normalizes the 'Intra_ID_Time_Gap' value by binning it into categories
fn intra_id_time_gap_norm(value: f64) -> i8 {
    if value.is_nan() {
        return -1;
    }
    
    if value <= 5.1 {
        0
    } else if value <= 10.1 {
        1
    } else if value <= 20.1 {
        2
    } else if value <= 30.1 {
        3
    } else if value <= 40.1 {
        4
    } else if value <= 50.1 {
        5
    } else if value <= 2010.1 {
        6
    } else if value <= 5010.1 {
        7
    } else {
        8
    }
}

/// Normalizes the 'Time_Delta' value by binning it into categories
fn time_delta_time_gap_norm(value: f64) -> i8 {
    if value.is_nan() {
        return -1;
    }
    
    if value <= 0.05 {
        0
    } else if value <= 0.1 {
        1
    } else if value <= 0.2 {
        2
    } else if value <= 0.3 {
        3
    } else if value <= 0.4 {
        4
    } else if value <= 0.5 {
        5
    } else {
        6
    }
}

// ==============================================================================
// SECTION 2: CORE DATA PREPROCESSING AND SEGMENTATION
// ==============================================================================

/// Loads and applies a full preprocessing pipeline to a single CSV file
fn load_preprocess_data(file_path: &Path) -> Result<DataFrame> {
    // Read CSV file
    let mut df = CsvReader::from_path(file_path)
        .context("Failed to read CSV file")?
        .has_header(true)
        .finish()
        .context("Failed to parse CSV")?;
    
    // Check if required columns exist
    if !df.get_column_names().contains(&"CAN_ID") || 
        !df.get_column_names().contains(&"Time_Offset") {
        anyhow::bail!("Required columns 'CAN_ID' or 'Time_Offset' not found");
    }
    
    // Feature Engineering
    normalize_can_id_by_frequency(&mut df, "CAN_ID")?;
    
    // Calculate Time_Delta (difference between consecutive Time_Offset values)
    let time_offset = df.column("Time_Offset")?.f64()?;
    let time_delta: Vec<Option<f64>> = std::iter::once(None)
        .chain(time_offset.into_iter().zip(time_offset.into_iter().skip(1))
            .map(|(prev, curr)| {
                match (prev, curr) {
                    (Some(p), Some(c)) => Some(c - p),
                    _ => None,
                }
            }))
        .collect();
    
    df.with_column(Series::new("Time_Delta", time_delta))?;
    
    // Calculate Intra_ID_Time_Gap (difference within same CAN_ID)
    let intra_gap = df
        .clone()
        .lazy()
        .with_column(
            col("Time_Offset")
                .diff(1, Default::default())
                .over([col("CAN_ID")])
                .alias("Intra_ID_Time_Gap")
        )
        .collect()?;
    
    df = intra_gap;
    
    // Drop rows with null values in key columns
    df = df
        .lazy()
        .drop_nulls(Some(vec![
            col("Intra_ID_Time_Gap"),
            col("Time_Delta"),
        ]))
        .collect()?;
    
    // Apply normalization functions
    let intra_gap_col = df.column("Intra_ID_Time_Gap")?.f64()?;
    let intra_gap_norm: Vec<i8> = intra_gap_col
        .into_iter()
        .map(|v| v.map(intra_id_time_gap_norm).unwrap_or(-1))
        .collect();
    
    let time_delta_col = df.column("Time_Delta")?.f64()?;
    let time_delta_norm: Vec<i8> = time_delta_col
        .into_iter()
        .map(|v| v.map(time_delta_time_gap_norm).unwrap_or(-1))
        .collect();
    
    df.with_column(Series::new("Intra_ID_Time_Gap_Norm", intra_gap_norm))?;
    df.with_column(Series::new("Time_Delta_Norm", time_delta_norm))?;
    
    // Ensure Label column exists
    if !df.get_column_names().contains(&"Label") {
        let label_series = Series::new("Label", vec!["Normal"; df.height()]);
        df.with_column(label_series)?;
    }
    
    Ok(df)
}

/// Loads, preprocesses, and segments data from a single file
pub fn segment_from_file(
    data_directory: &Path,
    filename: &str,
    time_gap: f64,
) -> Result<(Vec<Vec<[i32; 2]>>, Vec<u8>)> {
    let full_path = data_directory.join(filename);
    let df = load_preprocess_data(&full_path)?;
    
    if df.height() == 0 {
        return Ok((vec![], vec![]));
    }
    
    // Extract required columns
    let time_offset = df.column("Time_Offset")?.f64()?;
    let time_delta_norm = df.column("Time_Delta_Norm")?.i8()?;
    let intra_gap_norm = df.column("Intra_ID_Time_Gap_Norm")?.i8()?;
    let labels = df.column("Label")?.utf8()?;
    
    let mut chunks: Vec<Vec<[i32; 2]>> = Vec::new();
    let mut segment_labels: Vec<u8> = Vec::new();
    
    // Get min and max time offsets
    let min_time = time_offset.min().context("Empty time_offset column")?;
    let max_time = time_offset.max().context("Empty time_offset column")?;
    
    let mut window_start = min_time;
    
    while window_start <= max_time {
        let window_end = window_start + time_gap;
        
        // Create segment mask
        let mut segment_features: Vec<[i32; 2]> = Vec::new();
        let mut has_attack = false;
        
        for i in 0..df.height() {
            if let Some(t) = time_offset.get(i) {
                if t >= window_start && t < window_end {
                    let td = time_delta_norm.get(i).unwrap_or(-1) as i32;
                    let ig = intra_gap_norm.get(i).unwrap_or(-1) as i32;
                    segment_features.push([td, ig]);
                    
                    if let Some(label) = labels.get(i) {
                        if label != "Normal" {
                            has_attack = true;
                        }
                    }
                }
            }
        }
        
        if !segment_features.is_empty() {
            chunks.push(segment_features);
            segment_labels.push(if has_attack { 1 } else { 0 });
        }
        
        window_start += time_gap;
    }
    
    Ok((chunks, segment_labels))
}

// ==============================================================================
// EXAMPLE USAGE
// ==============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_segment_from_file() {
        let data_dir = Path::new("./test_data");
        let filename = "test.csv";
        let time_gap = 100.0;
        
        match segment_from_file(data_dir, filename, time_gap) {
            Ok((chunks, labels)) => {
                println!("Processed {} segments", chunks.len());
                println!("Labels: {:?}", labels);
            }
            Err(e) => eprintln!("Error: {}", e),
        }
    }
}

fn main() -> Result<()> {
    // Example usage
    let data_directory = Path::new("/home/lisa/Arupreza/UIDS-II/Input_data/Kia");
    let filename = "example.csv";
    let time_gap = 100.0;
    
    let (chunks, labels) = segment_from_file(data_directory, filename, time_gap)?;
    
    println!("Total segments processed: {}", chunks.len());
    println!("First 5 labels: {:?}", &labels[..5.min(labels.len())]);
    
    Ok(())
}