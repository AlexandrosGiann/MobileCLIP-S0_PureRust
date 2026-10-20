//! Loader for the compiled `mobileclip-fp32-plan-v1` graph and its weights.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde_json::{Map, Value};

use crate::error::{bail, Error, Result};
use crate::tensor::Tensor;

pub const PLAN_FORMAT: &str = "mobileclip-fp32-plan-v1";

/// Node attributes, kept as JSON values.
#[derive(Clone, Debug, Default)]
pub struct Attrs(Map<String, Value>);

impl Attrs {
    pub fn int(&self, name: &str) -> Option<i64> {
        self.0.get(name).and_then(Value::as_i64)
    }

    pub fn int_or(&self, name: &str, default: i64) -> i64 {
        self.int(name).unwrap_or(default)
    }

    pub fn float(&self, name: &str) -> Option<f64> {
        self.0.get(name).and_then(Value::as_f64)
    }

    pub fn ints(&self, name: &str) -> Option<Vec<i64>> {
        self.0
            .get(name)?
            .as_array()?
            .iter()
            .map(Value::as_i64)
            .collect()
    }

    pub fn usizes(&self, name: &str) -> Option<Vec<usize>> {
        self.ints(name)
            .map(|v| v.into_iter().map(|x| x.max(0) as usize).collect())
    }

    pub fn str(&self, name: &str) -> Option<&str> {
        self.0.get(name).and_then(Value::as_str)
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    pub op: String,
    /// Empty strings are absent optional inputs.
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: Attrs,
    /// Compiler hint: pointwise / depthwise / grouped / generic / static_B / runtime_B.
    pub fast: Option<String>,
}

pub struct Plan {
    pub input_name: String,
    pub output_name: String,
    pub nodes: Vec<Node>,
    /// Static tensors (weights and folded constants) by name.
    pub statics: HashMap<String, Tensor>,
    pub weights_bytes: usize,
}

fn str_field(v: &Value, key: &str) -> Result<String> {
    v.get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| Error::new(format!("plan: missing string field '{}'", key)))
}

fn str_list(v: &Value, key: &str) -> Result<Vec<String>> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new(format!("plan: missing list '{}'", key)))?
        .iter()
        .map(|x| match x {
            Value::String(s) => Ok(s.clone()),
            Value::Null => Ok(String::new()),
            _ => Err(Error::new(format!("plan: bad entry in '{}'", key))),
        })
        .collect()
}

/// Reads `weights.bin`, or the `weights.partNN.bin` pieces in order.
pub fn read_weights(dir: &Path) -> Result<Vec<u8>> {
    let single = dir.join("weights.bin");
    if single.exists() {
        return Ok(fs::read(&single)?);
    }
    let mut buf = Vec::new();
    for i in 0.. {
        let part = dir.join(format!("weights.part{:02}.bin", i));
        if !part.exists() {
            if i == 0 {
                bail!(
                    "no weights.bin or weights.part00.bin in {}",
                    dir.display()
                );
            }
            break;
        }
        buf.extend_from_slice(&fs::read(&part)?);
    }
    Ok(buf)
}

impl Plan {
    /// Loads `plan.json` and the weights from `dir` (`compiled-fp32-v1`).
    pub fn load(dir: &Path) -> Result<Plan> {
        let text = fs::read_to_string(dir.join("plan.json"))
            .map_err(|e| Error::new(format!("{}: {}", dir.join("plan.json").display(), e)))?;
        let weights = read_weights(dir)?;
        Plan::parse(&text, &weights)
    }

    pub fn parse(plan_json: &str, weights: &[u8]) -> Result<Plan> {
        let root: Value = serde_json::from_str(plan_json)?;

        if root.get("format").and_then(Value::as_str) != Some(PLAN_FORMAT) {
            bail!("plan: format is not {}", PLAN_FORMAT);
        }

        let first = |key: &str| -> Result<String> {
            let arr = root
                .get(key)
                .and_then(Value::as_array)
                .filter(|a| a.len() == 1)
                .ok_or_else(|| Error::new(format!("plan: expected one entry in '{}'", key)))?;
            str_field(&arr[0], "name")
        };
        let input_name = first("inputs")?;
        let output_name = first("outputs")?;

        if let Some(expected) = root
            .get("stats")
            .and_then(|s| s.get("weights_bytes"))
            .and_then(Value::as_u64)
        {
            if expected as usize != weights.len() {
                bail!(
                    "weights size mismatch: {} bytes, plan expects {}",
                    weights.len(),
                    expected
                );
            }
        }

        let mut statics = HashMap::new();
        let tensors = root
            .get("tensors")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::new("plan: missing 'tensors'"))?;

        for (name, info) in tensors {
            let offset = info.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
            let nbytes = info.get("nbytes").and_then(Value::as_u64).unwrap_or(0) as usize;
            let shape: Vec<usize> = info
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::new(format!("tensor {}: missing shape", name)))?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize))
                .collect::<Option<_>>()
                .ok_or_else(|| Error::new(format!("tensor {}: bad shape", name)))?;
            let dtype = info.get("dtype").and_then(Value::as_str).unwrap_or("");

            if offset + nbytes > weights.len() || nbytes % 4 != 0 {
                bail!("tensor {}: bytes {}+{} out of range", name, offset, nbytes);
            }
            let bytes = &weights[offset..offset + nbytes];
            let words = bytes
                .chunks_exact(4)
                .map(|c| [c[0], c[1], c[2], c[3]]);

            let n: usize = shape.iter().product();
            if n * 4 != nbytes {
                bail!("tensor {}: shape {:?} does not match {} bytes", name, shape, nbytes);
            }

            let t = match dtype {
                "f32" => Tensor::f32(words.map(f32::from_le_bytes).collect(), shape),
                "i32" => Tensor::i32(words.map(i32::from_le_bytes).collect(), shape),
                _ => bail!("tensor {}: unsupported dtype '{}'", name, dtype),
            };
            statics.insert(name.clone(), t);
        }

        let mut nodes = Vec::new();
        for n in root
            .get("nodes")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::new("plan: missing 'nodes'"))?
        {
            nodes.push(Node {
                name: str_field(n, "name").unwrap_or_default(),
                op: str_field(n, "op")?,
                inputs: str_list(n, "inputs")?,
                outputs: str_list(n, "outputs")?,
                attrs: Attrs(
                    n.get("attrs")
                        .and_then(Value::as_object)
                        .cloned()
                        .unwrap_or_default(),
                ),
                fast: n
                    .get("fast")
                    .and_then(|f| f.get("kind"))
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            });
        }

        Ok(Plan {
            input_name,
            output_name,
            nodes,
            statics,
            weights_bytes: weights.len(),
        })
    }
}
