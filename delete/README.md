# Delete Folder

This folder contains files and directories that were moved during codebase cleanup on October 3, 2025.

## Files Moved Here

### Temporary Database Files
- `*.db` - Demo database artifacts created during testing

### Development Documentation
- `*SUMMARY.md` - Development progress summaries
- `*IMPROVEMENTS*.md` - Improvement documentation
- `*FIXES*.md` - Fix documentation  
- `*COMPLETE*.md` - Completion documentation
- `FINAL_POLISH_SHIP_READY.md` - Final development documentation

### Temporary Cryptographic Materials
- `demo_keys/` - Temporary demo keys directory
- `assurance_keys/` - Temporary assurance keys directory

### Temporary Storage/Queue Directories
- `deferred_lcm_queue/` - Temporary LCM queue storage
- `deferred_lcm_storage/` - Temporary LCM storage

### Build Artifacts & Cache
- `__pycache__/` - Python bytecode cache
- `build_and_deploy.py` - Superseded build script

### Standalone Test Files
- `test_ultimate_wrapper.py` - Standalone test file moved to proper test organization

## Rationale

These files were moved to clean up the repository and separate:
- **Production code** (kept in main directories)
- **Examples/demos** (moved to `examples/` directory)
- **Development artifacts** (moved here for potential deletion)
- **Temporary files** (moved here for cleanup)

## Safe to Delete

All files in this directory are safe to delete as they are either:
1. Temporary artifacts that can be regenerated
2. Development documentation that served its purpose
3. Cache files that will be recreated as needed
4. Duplicate or superseded files

## Evidence Pack Files

Note: Evidence pack files (anchor.json, proof.json, etc.) were moved to `examples/evidence_pack/` rather than deleted, as they serve as examples of the system's output.