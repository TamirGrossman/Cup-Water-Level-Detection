version 2 of trying to fix the vaildtion and kfold using 5x check for each fold and separating the check between cup and fill % for each cup
make sure to use the vaildtion clean or model vaildation or both

One thing worth deciding deliberately: there are now two CV systems in the codebase — the old Model_Validation.py (leaky, fill-only stratified, single split) and the new Cross_Validation_Clean.py (leak-free, cup-balanced, repeated).
