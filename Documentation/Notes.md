# Notes and Comments
- is multiscale reconstruction loss the right way to do it?
    - I learned that the higher the model dim the worse the wavelet predictions are.
    - I added this to maybe steer the model into the right direction
- I want to do another huge overhaul to make everything more modular so we can test hyperparams easier
- Needs better documentation
- I need comments on the everything. model,trainer, sampler. I think theres a fundamental mistake in the pipeline somewhere. 
- model kind of already serves as a multi dim prediction with each wavelet band as a dim.
    - predictions are REALLY BAD
- I'm not sure how the returns look somewhat okay since the wavelet predictions are so bad
- I have a feeling the model architecture/tensor stuff is the issue
    - I just realized maybe the earlier iteration of the time series had "better" is because it predicted returns rather than raw price.
        - since returns are cum prod, obviously it returns a cone shape prediction space



# Plans for improvement:
- more granular documentation
- dynamic history lengths
- dynamic scheduler for noise, dropouts, everything
    - probably needs a function/class to pass it through
- the big overhaul should allow the config to be better documented and utilized. it is confusing
- VAE?
- "heat map" visualization for predictions
- box plot for each timestep