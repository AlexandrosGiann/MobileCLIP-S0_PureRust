//! Graph executor for the compiled plan.
//!
//! Tensor names are resolved to slots once. A run walks the nodes in plan
//! order; every value is dropped after its last reader, and a reader that
//! receives the last reference to a buffer may overwrite it in place.
//! `Div -> Erf -> Add -> Mul -> Mul` GELU chains whose intermediates have
//! no other reader run as one fused kernel.

use std::collections::HashMap;
use std::time::Instant;

use crate::error::{bail, Error, Result};
use crate::kernels::conv::{conv2d, ConvParams};
use crate::kernels::ops::{self, BinOp};
use crate::par::Par;
use crate::plan::{Attrs, Plan};
use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    Add,
    AveragePool,
    BatchNormalization,
    Concat,
    Conv,
    Div,
    Erf,
    Gather,
    MatMul,
    Mul,
    Pad,
    ReduceMean,
    Relu,
    Reshape,
    Shape,
    Sigmoid,
    Slice,
    Softmax,
    Split,
    Squeeze,
    Sub,
    Transpose,
    Unsqueeze,
    /// Fused GELU chain (not a plan operator).
    Gelu,
}

impl Op {
    fn parse(s: &str) -> Option<Op> {
        Some(match s {
            "Add" => Op::Add,
            "AveragePool" => Op::AveragePool,
            "BatchNormalization" => Op::BatchNormalization,
            "Concat" => Op::Concat,
            "Conv" => Op::Conv,
            "Div" => Op::Div,
            "Erf" => Op::Erf,
            "Gather" => Op::Gather,
            "MatMul" => Op::MatMul,
            "Mul" => Op::Mul,
            "Pad" => Op::Pad,
            "ReduceMean" => Op::ReduceMean,
            "Relu" => Op::Relu,
            "Reshape" => Op::Reshape,
            "Shape" => Op::Shape,
            "Sigmoid" => Op::Sigmoid,
            "Slice" => Op::Slice,
            "Softmax" => Op::Softmax,
            "Split" => Op::Split,
            "Squeeze" => Op::Squeeze,
            "Sub" => Op::Sub,
            "Transpose" => Op::Transpose,
            "Unsqueeze" => Op::Unsqueeze,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Op::Add => "Add",
            Op::AveragePool => "AveragePool",
            Op::BatchNormalization => "BatchNormalization",
            Op::Concat => "Concat",
            Op::Conv => "Conv",
            Op::Div => "Div",
            Op::Erf => "Erf",
            Op::Gather => "Gather",
            Op::MatMul => "MatMul",
            Op::Mul => "Mul",
            Op::Pad => "Pad",
            Op::ReduceMean => "ReduceMean",
            Op::Relu => "Relu",
            Op::Reshape => "Reshape",
            Op::Shape => "Shape",
            Op::Sigmoid => "Sigmoid",
            Op::Slice => "Slice",
            Op::Softmax => "Softmax",
            Op::Split => "Split",
            Op::Squeeze => "Squeeze",
            Op::Sub => "Sub",
            Op::Transpose => "Transpose",
            Op::Unsqueeze => "Unsqueeze",
            Op::Gelu => "Gelu",
        }
    }
}

/// Where a node input comes from.
#[derive(Clone, Copy, Debug)]
enum Src {
    None,
    Static(usize),
    /// A runtime value; `uses` is how many references this input
    /// accounts for (2 for x of a fused GELU, which the chain reads twice).
    Value { slot: usize, uses: u32 },
}

struct Step {
    op: Op,
    name: String,
    inputs: Vec<Src>,
    outputs: Vec<usize>,
    attrs: Attrs,
    /// Constant scalars of a fused GELU: div, add, mul.
    gelu: [f32; 3],
}

/// Per-operator time of one or more runs.
#[derive(Default, Debug, Clone)]
pub struct Profile {
    pub ops: HashMap<&'static str, (u32, f64)>,
    pub total_ms: f64,
    /// Print every step's time and output shape to stderr.
    pub trace: bool,
}

impl Profile {
    /// (operator, calls, milliseconds), slowest first.
    pub fn sorted(&self) -> Vec<(&'static str, u32, f64)> {
        let mut v: Vec<_> = self.ops.iter().map(|(k, &(c, t))| (*k, c, t)).collect();
        v.sort_by(|a, b| b.2.total_cmp(&a.2));
        v
    }
}

pub struct Executor {
    steps: Vec<Step>,
    statics: Vec<Tensor>,
    slots: usize,
    input_slot: usize,
    output_slot: usize,
    base_uses: Vec<u32>,
    par: Par,
    plan_nodes: usize,
    weights_bytes: usize,
}

impl Executor {
    pub fn new(plan: Plan, par: Par) -> Result<Executor> {
        let Plan { input_name, output_name, nodes, statics: static_map, weights_bytes } = plan;

        let mut statics = Vec::new();
        let mut static_ids: HashMap<String, usize> = HashMap::new();
        let mut slot_ids: HashMap<String, usize> = HashMap::new();
        let input_slot = 0;
        slot_ids.insert(input_name.clone(), input_slot);

        for n in &nodes {
            if Op::parse(&n.op).is_none() {
                bail!("unsupported operator '{}' at node {}", n.op, n.name);
            }
            for o in &n.outputs {
                let next = slot_ids.len();
                slot_ids.entry(o.clone()).or_insert(next);
            }
        }
        let output_slot = *slot_ids
            .get(&output_name)
            .ok_or_else(|| Error::new(format!("output '{}' is never produced", output_name)))?;

        // Uses per runtime value, counted per input occurrence.
        let mut base_uses = vec![0u32; slot_ids.len()];
        for n in &nodes {
            for i in &n.inputs {
                if let Some(&s) = slot_ids.get(i) {
                    base_uses[s] += 1;
                }
            }
        }

        let mut static_map = static_map;
        let mut resolve = |name: &str, statics: &mut Vec<Tensor>| -> Result<Src> {
            if name.is_empty() {
                return Ok(Src::None);
            }
            if let Some(&s) = slot_ids.get(name) {
                return Ok(Src::Value { slot: s, uses: 1 });
            }
            if let Some(&i) = static_ids.get(name) {
                return Ok(Src::Static(i));
            }
            let t = static_map
                .remove(name)
                .ok_or_else(|| Error::new(format!("tensor '{}' is neither produced nor static", name)))?;
            statics.push(t);
            static_ids.insert(name.to_owned(), statics.len() - 1);
            Ok(Src::Static(statics.len() - 1))
        };

        let mut steps = Vec::with_capacity(nodes.len());
        let mut i = 0;
        while i < nodes.len() {
            if let Some(chain) = gelu_chain(&nodes, i, &slot_ids, &base_uses) {
                let x = Src::Value { slot: slot_ids[&nodes[i].inputs[0]], uses: 2 };
                let mut scalars = [0.0f32; 3];
                for (k, name) in [&nodes[i].inputs[1], &chain.add, &chain.mul].into_iter().enumerate() {
                    let src = resolve(name, &mut statics)?;
                    let v = match src {
                        Src::Static(si) if statics[si].len() == 1 && statics[si].is_f32() => {
                            statics[si].as_f32()?[0]
                        }
                        _ => bail!("GELU chain at node {} has a non-constant scalar", nodes[i].name),
                    };
                    scalars[k] = v;
                }
                steps.push(Step {
                    op: Op::Gelu,
                    name: nodes[i].name.clone(),
                    inputs: vec![x],
                    outputs: vec![slot_ids[&nodes[i + 4].outputs[0]]],
                    attrs: Attrs::default(),
                    gelu: scalars,
                });
                i += 5;
                continue;
            }

            let n = &nodes[i];
            let inputs = n
                .inputs
                .iter()
                .map(|name| resolve(name, &mut statics))
                .collect::<Result<Vec<_>>>()?;
            steps.push(Step {
                op: Op::parse(&n.op).unwrap(),
                name: n.name.clone(),
                inputs,
                outputs: n.outputs.iter().map(|o| slot_ids[o]).collect(),
                attrs: n.attrs.clone(),
                gelu: [0.0; 3],
            });
            i += 1;
        }

        Ok(Executor {
            steps,
            statics,
            slots: slot_ids.len(),
            input_slot,
            output_slot,
            base_uses,
            par,
            plan_nodes: nodes.len(),
            weights_bytes,
        })
    }

    pub fn threads(&self) -> usize {
        self.par.threads
    }

    pub fn plan_nodes(&self) -> usize {
        self.plan_nodes
    }

    /// Kernel launches per run, after GELU fusion.
    pub fn steps(&self) -> usize {
        self.steps.len()
    }

    pub fn fused_gelus(&self) -> usize {
        self.steps.iter().filter(|s| s.op == Op::Gelu).count()
    }

    pub fn weights_bytes(&self) -> usize {
        self.weights_bytes
    }

    /// Runs the graph on one input and returns the f32 output buffer.
    pub fn run(&self, input: Vec<f32>, shape: &[usize], mut profile: Option<&mut Profile>) -> Result<Vec<f32>> {
        let t_run = Instant::now();
        let mut env: Vec<Option<Tensor>> = vec![None; self.slots];
        let mut uses = self.base_uses.clone();
        env[self.input_slot] = Some(Tensor::f32(input, shape.to_vec()));

        for (si, step) in self.steps.iter().enumerate() {
            let t0 = Instant::now();
            let mut ins: Vec<Option<Tensor>> = Vec::with_capacity(step.inputs.len());
            for src in &step.inputs {
                ins.push(match *src {
                    Src::None => None,
                    Src::Static(i) => Some(self.statics[i].clone()),
                    Src::Value { slot, uses: n } => {
                        uses[slot] -= n;
                        let last = uses[slot] == 0 && slot != self.output_slot;
                        let t = if last { env[slot].take() } else { env[slot].clone() };
                        Some(t.ok_or_else(|| {
                            Error::new(format!("step {} ({}): input not computed", si, step.name))
                        })?)
                    }
                });
            }

            let outs = self
                .exec(step, ins)
                .map_err(|e| e.context(format!("node '{}' ({})", step.name, step.op.name())))?;
            if outs.len() != step.outputs.len() {
                bail!("node '{}' produced {} outputs, expected {}", step.name, outs.len(), step.outputs.len());
            }
            for (&slot, t) in step.outputs.iter().zip(outs) {
                // Values nobody reads (and that are not the output) are dropped now.
                if uses[slot] > 0 || slot == self.output_slot {
                    env[slot] = Some(t);
                }
            }

            if let Some(p) = profile.as_deref_mut() {
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                let e = p.ops.entry(step.op.name()).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += ms;
                if p.trace {
                    let shape = step.outputs.first().and_then(|&s| env[s].as_ref()).map(|t| t.shape.clone());
                    eprintln!("{:>4} {:>8.3} ms  {:<10} {:?}  {}", si, ms, step.op.name(), shape.unwrap_or_default(), step.name);
                }
            }
        }

        if let Some(p) = profile {
            p.total_ms += t_run.elapsed().as_secs_f64() * 1e3;
        }

        let out = env[self.output_slot]
            .take()
            .ok_or_else(|| Error::new("graph output was not produced"))?;
        Ok(out.into_f32_vec())
    }

    fn exec(&self, step: &Step, mut ins: Vec<Option<Tensor>>) -> Result<Vec<Tensor>> {
        let par = &self.par;
        let a = &step.attrs;
        let mut take = |i: usize| -> Result<Tensor> {
            ins.get_mut(i)
                .and_then(Option::take)
                .ok_or_else(|| Error::new(format!("missing input {}", i)))
        };

        let one = |t: Tensor| Ok(vec![t]);

        match step.op {
            Op::Gelu => {
                let [div, add, mul] = step.gelu;
                one(ops::gelu(par, take(0)?, div, add, mul))
            }
            Op::Conv => {
                let x = take(0)?;
                let w = take(1)?;
                let b = take(2).ok();
                let ws = &w.shape;
                let pads = a.usizes("pads").unwrap_or_else(|| vec![0; 4]);
                let strides = a.usizes("strides").unwrap_or_else(|| vec![1, 1]);
                let dil = a.usizes("dilations").unwrap_or_else(|| vec![1, 1]);
                if a.str("auto_pad").is_some_and(|s| s != "NOTSET") {
                    bail!("Conv auto_pad is not supported");
                }
                if pads.len() != 4 || strides.len() != 2 || dil.len() != 2 {
                    bail!("only 2-D Conv is supported");
                }
                let p = ConvParams {
                    group: a.int_or("group", 1) as usize,
                    pads: [pads[0], pads[1], pads[2], pads[3]],
                    strides: [strides[0], strides[1]],
                    dilations: [dil[0], dil[1]],
                };
                let bias = match &b {
                    Some(b) => Some(b.as_f32()?),
                    None => None,
                };
                let (out, shape) = conv2d(par, x.as_f32()?, &x.shape, w.as_f32()?, ws, bias, &p)?;
                one(Tensor::f32(out, shape))
            }
            Op::MatMul => {
                let x = take(0)?;
                let y = take(1)?;
                one(ops::matmul(par, &x, &y)?)
            }
            Op::Add => one(ops::binary(take(0)?, take(1)?, BinOp::Add)?),
            Op::Sub => one(ops::binary(take(0)?, take(1)?, BinOp::Sub)?),
            Op::Mul => one(ops::binary(take(0)?, take(1)?, BinOp::Mul)?),
            Op::Div => one(ops::binary(take(0)?, take(1)?, BinOp::Div)?),
            Op::Relu => one(ops::relu(take(0)?)),
            Op::Sigmoid => one(ops::sigmoid(take(0)?)),
            Op::Erf => one(ops::erf(take(0)?)),
            Op::Shape => one(ops::shape_of(&take(0)?)),
            Op::Reshape => {
                let x = take(0)?;
                let spec = take(1)?.to_i64_vec();
                one(ops::reshape(&x, &spec, a.int_or("allowzero", 0) != 0)?)
            }
            Op::Unsqueeze => one(ops::unsqueeze(&take(0)?, &a.ints("axes").unwrap_or_default())?),
            Op::Squeeze => one(ops::squeeze(&take(0)?, &a.ints("axes").unwrap_or_default())?),
            Op::Transpose => one(ops::transpose(&take(0)?, a.ints("perm").as_deref())?),
            Op::Concat => {
                let xs: Vec<Tensor> = ins.into_iter().flatten().collect();
                one(ops::concat(&xs, a.int_or("axis", 0))?)
            }
            Op::Gather => {
                let d = take(0)?;
                let idx = take(1)?;
                one(ops::gather(&d, &idx, a.int_or("axis", 0))?)
            }
            Op::Slice => {
                let d = take(0)?;
                let starts = take(1)?.to_i64_vec();
                let ends = take(2)?.to_i64_vec();
                let axes = take(3).ok().map(|t| t.to_i64_vec());
                let steps = take(4).ok().map(|t| t.to_i64_vec());
                one(ops::slice(&d, &starts, &ends, axes.as_deref(), steps.as_deref())?)
            }
            Op::Split => {
                let x = take(0)?;
                let sizes = a.usizes("split");
                ops::split(&x, a.int_or("axis", 0), sizes.as_deref(), step.outputs.len())
            }
            Op::ReduceMean => one(ops::reduce_mean(
                &take(0)?,
                &a.ints("axes").unwrap_or_default(),
                a.int_or("keepdims", 1) != 0,
            )?),
            Op::Softmax => one(ops::softmax(take(0)?, a.int_or("axis", -1))?),
            Op::BatchNormalization => {
                let x = take(0)?;
                let (s, b, m, v) = (take(1)?, take(2)?, take(3)?, take(4)?);
                let eps = a.float("epsilon").unwrap_or(1e-5) as f32;
                one(ops::batch_norm(x, s.as_f32()?, b.as_f32()?, m.as_f32()?, v.as_f32()?, eps)?)
            }
            Op::Pad => {
                if a.str("mode").is_some_and(|m| m != "constant") {
                    bail!("only constant Pad is supported");
                }
                let x = take(0)?;
                let pads = take(1)?.to_i64_vec();
                let value = match take(2) {
                    Ok(v) => v.as_f32()?.first().copied().unwrap_or(0.0),
                    Err(_) => 0.0,
                };
                one(ops::pad_constant(x, &pads, value)?)
            }
            Op::AveragePool => {
                if a.int_or("ceil_mode", 0) != 0 {
                    bail!("AveragePool ceil_mode is not supported");
                }
                let k = a.usizes("kernel_shape").unwrap_or_default();
                if k.len() != 2 {
                    bail!("only 2-D AveragePool is supported");
                }
                let s = a.usizes("strides").unwrap_or_else(|| k.clone());
                let p = a.usizes("pads").unwrap_or_else(|| vec![0; 4]);
                one(ops::average_pool(
                    &take(0)?,
                    [k[0], k[1]],
                    [s[0], s[1]],
                    [p[0], p[1], p[2], p[3]],
                    a.int_or("count_include_pad", 0) != 0,
                )?)
            }
        }
    }
}

struct GeluChain {
    add: String,
    mul: String,
}

/// Recognises `Div -> Erf -> Add -> Mul(x, .) -> Mul` at node `i` whose
/// intermediates have no reader outside the chain.
fn gelu_chain(
    nodes: &[crate::plan::Node],
    i: usize,
    slots: &HashMap<String, usize>,
    uses: &[u32],
) -> Option<GeluChain> {
    let c = nodes.get(i..i + 5)?;
    let ops = ["Div", "Erf", "Add", "Mul", "Mul"];
    if c.iter().zip(ops).any(|(n, op)| n.op != op || n.outputs.len() != 1) {
        return None;
    }
    let out = |k: usize| &c[k].outputs[0];
    let x = c[0].inputs.first()?;
    let reads = |k: usize, name: &String| c[k].inputs.iter().any(|s| s == name);
    if !reads(1, out(0)) || !reads(2, out(1)) || !reads(3, out(2)) || !reads(3, x) || !reads(4, out(3)) {
        return None;
    }
    if c[0].inputs.len() != 2 || c[2].inputs.len() != 2 || c[3].inputs.len() != 2 || c[4].inputs.len() != 2 {
        return None;
    }
    if !slots.contains_key(x) || (0..4).any(|k| uses[slots[out(k)]] != 1) {
        return None;
    }
    let other = |k: usize, produced: &String| c[k].inputs.iter().find(|s| *s != produced).cloned();
    Some(GeluChain {
        add: other(2, out(1))?,
        mul: other(4, out(3))?,
    })
}
