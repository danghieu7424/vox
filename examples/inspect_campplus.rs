use ort::session::Session;
use ndarray::Array3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let env = ort::init().with_name("test_campplus").build()?;
    let mut session = Session::builder(&env)?.commit_from_file("models/campplus.onnx")?;

    // Try (1, 100 frames, 80 mel features)
    let test_fbank = Array3::<f32>::zeros((1, 100, 80));
    let tensor = ort::value::Tensor::from_array(test_fbank)?;

    match session.run(ort::inputs!["x" => tensor]) {
        Ok(outputs) => {
            let (shape, data) = outputs["embedding"].try_extract_tensor::<f32>()?;
            println!("Success with shape (1, 100, 80)! Output embedding shape: {:?}, len: {}", shape, data.len());
        }
        Err(e) => {
            println!("Error with (1, 100, 80): {}", e);
        }
    }

    Ok(())
}
