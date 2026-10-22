//! Experimental coordinate channel. Loaded explicitly by the host probe only;
//! the production no-touch API retains its existing scores and acceptance rule.
use serde::Deserialize;

#[derive(Deserialize)]
struct KeyModel {
    mean: [f64; 2],
    inverse: [f64; 3],
    log_det: f64,
}

#[derive(Deserialize)]
pub struct TouchModel {
    schema: String,
    weight: f64,
    degrees_of_freedom: f64,
    max_log_ratio: f64,
    keys: Vec<KeyModel>,
}

impl TouchModel {
    pub fn from_json(json: &str) -> Result<Self, String> {
        let m: Self = serde_json::from_str(json).map_err(|e| e.to_string())?;
        if m.schema != "tsi-coordinate-v1" || m.keys.len() != 26 ||
            !m.weight.is_finite() || !(0. ..=2.).contains(&m.weight) ||
            m.degrees_of_freedom != 4. || !m.max_log_ratio.is_finite() ||
            !(0. ..=8.).contains(&m.max_log_ratio) || m.keys.iter().any(|k| {
                let [a,b,c] = k.inverse;
                !k.mean.iter().chain(k.inverse.iter()).all(|v|v.is_finite()) ||
                    !k.log_det.is_finite() || a <= 0. || c <= 0. || a*c-b*b <= 0.
            }) {
            return Err("Invalid TSI coordinate model".into());
        }
        Ok(m)
    }

    /// Geometry is [cx, cy, width, height] for a..z. Coordinates can be pixels
    /// or any common units: offsets are divided by each key's own dimensions.
    /// Missing/malformed evidence falls back to the existing text-only scorer.
    pub fn prepare(&self, typed: &str, geometry: &[f32], points: &[f32]) -> Option<TouchScorer> {
        if self.weight == 0. || typed.is_empty() || typed.len() > 32 ||
            !typed.bytes().all(|b| b.is_ascii_lowercase()) || geometry.len() != 104 ||
            points.len() != typed.len()*2 || geometry.chunks_exact(4).any(|g|
                !g.iter().all(|v|v.is_finite()) || g[2] <= 0. || g[3] <= 0.) {
            return None;
        }
        let scores = points.chunks_exact(2).map(|p| {
            if !p.iter().all(|v|v.is_finite()) {return None;}
            let mut row = [0.; 26];
            for (i, (g, k)) in geometry.chunks_exact(4).zip(&self.keys).enumerate() {
                let x = (p[0] as f64-g[0] as f64)/g[2] as f64-k.mean[0];
                let y = (p[1] as f64-g[1] as f64)/g[3] as f64-k.mean[1];
                let [a,b,c] = k.inverse;
                let q = (a*x*x+2.*b*x*y+c*y*y).max(0.);
                row[i] = -0.5*k.log_det-3.*(q/4.).ln_1p();
            }
            Some(row)
        }).collect();
        Some(TouchScorer { typed: typed.as_bytes().to_vec(), scores,
                           weight: self.weight as f32, cap: self.max_log_ratio })
    }
}

pub struct TouchScorer {
    typed: Vec<u8>,
    scores: Vec<Option<[f64; 26]>>,
    weight: f32,
    cap: f64,
}

impl TouchScorer {
    pub fn adjustment(&self, word: &str, edits: u8, completed: usize) -> f32 {
        let word = word.to_ascii_lowercase();
        if completed > 0 || word.len() != self.typed.len() ||
            !word.bytes().all(|b|b.is_ascii_lowercase()) ||
            word.bytes().zip(&self.typed).filter(|(a,b)|a != *b).count() != edits as usize {
            return 0.;
        }
        let mut ratio = 0.;
        for (i, (a,b)) in self.typed.iter().copied().zip(word.bytes()).enumerate() {
            if a == b {continue;}
            // An absent tap contributes no evidence, including at changed letters.
            if let Some(row) = self.scores[i] {
                ratio += row[(b-b'a') as usize]-row[(a-b'a') as usize];
            }
        }
        self.weight * ratio.clamp(-self.cap,self.cap) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn model() -> TouchModel {
        TouchModel::from_json(&serde_json::json!({"schema":"tsi-coordinate-v1", "weight":0.125,
            "degrees_of_freedom":4., "max_log_ratio":8.,
            "keys":(0..26).map(|_|serde_json::json!({"mean":[0.,0.],"inverse":[16.,0.,16.],"log_det":-5.545})).collect::<Vec<_>>()}).to_string()).unwrap()
    }
    fn geometry() -> Vec<f32> {(0..26).flat_map(|i|[i as f32,0.,1.,2.]).collect()}
    #[test] fn invariant_to_translation_and_independent_axis_scaling() {
        let m=model(); let g=geometry(); let p=vec![0.8, 0.2, 1., 0., 2., 0.];
        let a=m.prepare("abc",&g,&p).unwrap().adjustment("bbc",1,0);
        assert!(a>0.);
        let mapped_g:Vec<_>=g.chunks_exact(4).flat_map(|v|[v[0]*3.+10.,v[1]*7.-50.,v[2]*3.,v[3]*7.]).collect();
        let mapped_p:Vec<_>=p.chunks_exact(2).flat_map(|v|[v[0]*3.+10.,v[1]*7.-50.]).collect();
        let b=m.prepare("abc",&mapped_g,&mapped_p).unwrap().adjustment("bbc",1,0);
        assert!((a-b).abs()<1e-6);
    }
    #[test] fn missing_evidence_and_unsupported_edit_paths_add_nothing() {
        let m=model(); let g=geometry();
        assert!(m.prepare("abc",&[],&[]).is_none());
        let scorer=m.prepare("abc",&g,&[f32::NAN;6]).unwrap();
        assert_eq!(scorer.adjustment("bbc",1,0),0.);
        let scorer=m.prepare("abc",&g,&[0.;6]).unwrap();
        for (word,edits,completed) in [("ab",1,0),("bac",1,0),("abcd",0,1),("a'c",1,0)] {
            assert_eq!(scorer.adjustment(word,edits,completed),0.);
        }
        assert_eq!(scorer.adjustment("abc",0,0),0.);
    }
    #[test] fn rejects_invalid_covariance_and_geometry() {
        assert!(TouchModel::from_json("{}").is_err());
        let m=model(); let mut g=geometry();g[2]=0.;
        assert!(m.prepare("abc",&g,&[0.;6]).is_none());
    }
}
