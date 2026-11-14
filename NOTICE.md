# References and third-party software

This project uses PyTorch, NumPy, Sionna, Matplotlib, Gradio, PyYAML,
TensorBoard and tqdm through their Python APIs. Their respective copyrights
and licenses remain with their authors. The lockfiles identify the versions
resolved for the declared environments; dependencies are not vendored here.

The Transformer uses PyTorch's standard encoder/decoder modules. The channel
generator calls Sionna's CDL-A implementation. The coefficient-wise AR prior
is distinct from a full reproduction of the cited angular-delay PAD algorithm.
Literature references and modeling distinctions are listed in
[methodology](docs/methodology.md).

The inference examples are synthetic channels produced by the project's
generator. Trained weights and example data belong to the same maintained
experiment. Third-party papers, university administrative documents and the
original thesis report are not redistributed in this source package.
