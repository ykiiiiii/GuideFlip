"""ADFlip, vendored, as the structure-conditioned sequence prior.

Upstream: https://github.com/ykiiiiii/ADFLIP at daa01da.

Narrowed to one job: read a structure and the residues decided so far, return
logits over amino acids. The sampling ADFlip can do itself is not here, because
GuideFlip does its own -- see NOTICE for what was removed and why.
"""
