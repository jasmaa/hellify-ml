import onnxruntime as rt
import numpy as np
from PIL import Image

usage = """
Usage: python generate.py [input path]"""


def create_im(path):
    sess = rt.InferenceSession('./models/hellscape.onnx')

    # prepare image
    im = Image.open(path)
    im = im.resize((256, 256))
    in_data = np.array(im)
    in_data = np.rollaxis(in_data, 2)
    in_data = np.expand_dims(in_data, axis=0)
    in_data = in_data.astype(np.float32)

    input_name = sess.get_inputs()[0].name
    out_data = sess.run(None, {input_name: in_data})

    # Convert output to pil data
    converted = ((np.transpose(out_data[0].squeeze(), (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
    out_im = Image.fromarray(converted)
    name, suffix = path.split(".")
    out_im.save(f"{name}_out.{suffix}")
    print(f"Generated image: {name}_out.{suffix}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        create_im(sys.argv[1])
    else:
        print(usage)
