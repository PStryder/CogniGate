# CogniGate Test Suite

Comprehensive test suite for CogniGate with focus on security validation.

## Test Structure

```
tests/
├── conftest.py           # Pytest fixtures and configuration
├── test_security.py      # Security tests for BLOCKER fixes (CRITICAL)
├── test_prompt.py        # Prompt building and sanitization tests
├── test_plugins.py       # Plugin system tests
├── test_models.py        # Data model tests
└── __init__.py
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run with Coverage
```bash
pytest --cov=cognigate --cov-report=html
```

### Run Specific Test Files
```bash
# Security tests only
pytest tests/test_security.py

# Prompt tests only
pytest tests/test_prompt.py

# Plugin tests only
pytest tests/test_plugins.py
```

### Run by Marker
```bash
# Run only security tests
pytest -m security

# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"
```

## Test Categories

### Security Tests (`test_security.py`)
**CRITICAL** - Tests for the three BLOCKER vulnerabilities:

1. **BLOCKER-001**: Plugin system arbitrary code execution
   - Tests world-writable directory rejection
   - Tests group-writable directory rejection
   - Tests safe permissions acceptance

2. **BLOCKER-002**: Prompt injection prevention
   - Tests SYSTEM OVERRIDE sanitization
   - Tests IGNORE PREVIOUS sanitization
   - Tests special token removal
   - Tests XML delimiter separation
   - Tests constraint sanitization
   - Tests length limits

3. **BLOCKER-003**: Path traversal in FileSink
   - Tests task_id sanitization
   - Tests lease_id sanitization
   - Tests path resolution validation
   - Tests dots and slashes removal

### Prompt Tests (`test_prompt.py`)
Tests for prompt construction and sanitization:
- Prompt structure validation
- Content sanitization
- XML delimiter usage
- Constraint handling
- Length limits

### Plugin Tests (`test_plugins.py`)
Tests for plugin system:
- Plugin registration
- Plugin discovery
- Permission validation
- Error handling
- FileSink implementation

### Model Tests (`test_models.py`)
Tests for data models:
- Model creation
- Validation rules
- Serialization
- Field requirements

## Test Fixtures

### Provided by `conftest.py`:
- `temp_dir`: Temporary directory for tests
- `safe_temp_dir`: Directory with secure permissions (0o700)
- `unsafe_temp_dir`: Directory with insecure permissions (0o777)
- `sample_lease`: Valid lease for testing
- `malicious_lease`: Lease with injection attempts
- `instruction_profile`: Sample instruction profile
- `test_settings`: Test configuration
- `app_client`: FastAPI test client

## Coverage Requirements

Target coverage: **80%+ for security-critical code**

Priority coverage areas:
1. `prompt.py` - 100% (prompt injection prevention)
2. `plugins/base.py` - 100% (plugin security)
3. `plugins/builtin_sinks.py` - 100% (path traversal prevention)
4. `models.py` - 90%
5. Other modules - 70%+

## Adding New Tests

### For Security Issues:
1. Add test to `test_security.py`
2. Mark with `@pytest.mark.security`
3. Document which vulnerability it tests
4. Include both exploit attempt and verification

### For Features:
1. Add appropriate test file
2. Follow naming convention: `test_<feature>.py`
3. Use descriptive test names: `test_<what>_<expected>`
4. Add docstrings explaining what's tested

### Test Naming Convention:
```python
def test_<component>_<scenario>_<expected_result>():
    """Brief description of what this tests."""
    # Arrange
    # Act
    # Assert
```

## Dependencies

Required testing packages (in requirements.txt):
```
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
httpx>=0.24.0  # For FastAPI testing
```

## CI/CD Integration

### GitHub Actions Example:
```yaml
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest --cov=cognigate --cov-report=xml

- name: Check coverage
  run: |
    coverage report --fail-under=80
```

## Security Test Validation

**BEFORE OSS v1 RELEASE**, verify all security tests pass:

```bash
# Run security tests with verbose output
pytest tests/test_security.py -v

# Expected output:
# test_security.py::TestBlocker001PluginSecurity::test_plugin_directory_world_writable_rejected PASSED
# test_security.py::TestBlocker001PluginSecurity::test_plugin_directory_group_writable_rejected PASSED
# test_security.py::TestBlocker002PromptInjection::test_system_override_injection_sanitized PASSED
# test_security.py::TestBlocker002PromptInjection::test_ignore_previous_instructions_sanitized PASSED
# test_security.py::TestBlocker003PathTraversal::test_path_traversal_in_task_id_blocked PASSED
# ... and more
```

All security tests MUST pass before releasing.

## Debugging Failed Tests

### Verbose output:
```bash
pytest -vv
```

### Show print statements:
```bash
pytest -s
```

### Stop on first failure:
```bash
pytest -x
```

### Run specific test:
```bash
pytest tests/test_security.py::TestBlocker001PluginSecurity::test_plugin_directory_world_writable_rejected -v
```

## Known Issues

None currently. If a test fails, it indicates either:
1. A regression in security fixes
2. An environment-specific issue
3. A test that needs updating

Report test failures in GitHub Issues.

---

**Test Status**: ✅ Ready for OSS v1
**Coverage Target**: 80%+
**Security Tests**: CRITICAL - Must pass before release
