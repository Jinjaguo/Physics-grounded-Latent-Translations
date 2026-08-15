# Wave42 receding-horizon force scheduling

Wave41 showed that simply shrinking the force destroys redirection.  Wave42
keeps the learned force direction but compares online schedules: first-step
only, full horizon, geometric decay, and receding replan/recovery.  The
frozen VAE/decoder/F1/F2 remain unchanged and no future action is an input.
The experiment evaluates whether the failure is caused by applying a long
latent correction in one shot.  Continue unless success or Wave78.
