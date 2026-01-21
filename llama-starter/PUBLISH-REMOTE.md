# 🌐 Publish to PyPI - Install from Anywhere

## Step 1: Create PyPI Account

### TestPyPI (For Testing)
1. Go to https://test.pypi.org/account/register/
2. Verify your email
3. Enable 2FA
4. Create API token: https://test.pypi.org/manage/account/token/
   - Token name: `llama-rag-lib-upload`
   - Scope: `Entire account`
   - Copy the token (starts with `pypi-`)

### Production PyPI
1. Go to https://pypi.org/account/register/
2. Verify your email
3. Enable 2FA
4. Create API token: https://pypi.org/manage/account/token/
   - Token name: `llama-rag-lib`
   - Scope: `Entire account`
   - Copy the token

## Step 2: Install Upload Tools

```bash
pip install twine
```

## Step 3: Upload to TestPyPI (Test First!)

```bash
cd /Users/p0d061n/Documents/apps/llama-starter

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# When prompted:
# Username: __token__
# Password: pypi-YOUR_TESTPYPI_TOKEN_HERE
```

## Step 4: Test Installation from TestPyPI

```bash
# In another project or virtual environment
# Note: --extra-index-url allows pip to get dependencies from PyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ llama_rag_lib

# Test it
python -c "from llama_rag import KnowledgeBaseService; print('✅ Works!')"
```

## Step 5: Upload to Production PyPI

```bash
cd /Users/p0d061n/Documents/apps/llama-starter

# Upload to PyPI
twine upload dist/*

# When prompted:
# Username: __token__
# Password: pypi-YOUR_PYPI_TOKEN_HERE
```

## Step 6: Install from Anywhere! 🎉

```bash
# Now anyone can install it
pip install llama-rag-lib

# Use it
python -c "from llama_rag import KnowledgeBaseService; print('✅ Installed from PyPI!')"
```

## Alternative: Save Credentials (Optional)

Create `~/.pypirc` to avoid entering tokens each time:

```bash
cat > ~/.pypirc << 'EOF'
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_PRODUCTION_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE
EOF

chmod 600 ~/.pypirc
```

Then you can upload without entering credentials:

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## Quick Command Reference

```bash
# Install twine
pip install twine

rm -rf dist/ build/ src/*.egg-info

# Build package
cd /Users/p0d061n/Documents/apps/llama-starter
python setup.py sdist

# Upload to TestPyPI
twine upload --repository testpypi dist/llama_rag_lib-0.1.0.tar.gz

# Test install (note: --extra-index-url for dependencies)
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ llama_rag_lib

pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --upgrade --no-cache-dir llama_rag_lib

# Upload to PyPI (production)
twine upload dist/llama_rag_lib-0.1.0.tar.gz

# Install from PyPI
pip install llama_rag_lib
```

## Using Makefile (Easier)

```bash
cd /Users/p0d061n/Documents/apps/llama-starter

# Install twine first
pip install twine

# Upload to TestPyPI
make publish-test

# Upload to PyPI
make publish
```

## Troubleshooting

### "Invalid or non-existent authentication"
**Solution:** Make sure you're using `__token__` as username and your actual token as password.

### "File already exists"
**Solution:** You need to bump the version:
```bash
make bump-patch  # 0.1.0 → 0.1.1
python setup.py sdist
twine upload dist/llama-rag-lib-0.1.1.tar.gz
```

### "403 Forbidden"
**Solution:** Check your token has correct permissions and hasn't expired.

## After Publishing

Your package will be available at:
- TestPyPI: https://test.pypi.org/project/llama-rag-lib/
- PyPI: https://pypi.org/project/llama-rag-lib/

Anyone can install it:
```bash
pip install llama-rag-lib
```

## Update Your Package Name (If Needed)

If `llama-rag-lib` is already taken on PyPI, change it in:
- `setup.py` → `name="your-unique-name"`
- `pyproject.toml` → `name = "your-unique-name"`

Then rebuild and upload.

---

## 🚀 Quick Start Publishing

```bash
# 1. Install twine
pip install twine

# 2. Get PyPI token from https://test.pypi.org/manage/account/token/

# 3. Upload to TestPyPI
cd /Users/p0d061n/Documents/apps/llama-starter
twine upload --repository testpypi dist/*
# Username: __token__
# Password: <paste your token>

# 4. Test it works
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ llama_rag_lib

# 5. If good, upload to production PyPI
# Get token from https://pypi.org/manage/account/token/
twine upload dist/*

# 6. Done! Install from anywhere:
pip install llama-rag-lib
```
