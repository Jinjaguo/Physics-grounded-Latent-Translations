# Wave 27 decoder-loss audit

The frozen representation decoder remained differentiable with respect to predicted latents. See `wave27_decoder_gradient_unit_test.json`; transition gradients were nonzero and representation gradients were zero. No cycle projection was used.
