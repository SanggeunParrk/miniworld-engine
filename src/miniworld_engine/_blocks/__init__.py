"""Model blocks the engine itself still builds (internal): the bundle coverage driver runs them, and ``modules.InputFeatureEmbedder`` holds the
SWA atom transformer. The model code that used to live here (diffusion processes, geometry helpers, layers, checkpoint adapters) is not part of
the engine; it is in git history before the commit that renamed this package."""
