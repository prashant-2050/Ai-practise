# GitHub Actions Deployment Guide

## 🚀 Auto-Deploy to Hugging Face Spaces

This guide shows you how to set up automatic deployment of your LLM to Hugging Face Spaces using GitHub Actions. Every time you push to the main branch, your demo will automatically update!

## 📋 Prerequisites

1. **GitHub Repository** (you have this ✅)
2. **Hugging Face Account** - [Sign up free here](https://huggingface.co/join)
3. **Hugging Face Access Token** - We'll create this below

## 🔧 Setup Steps

### Step 1: Create Hugging Face Account & Token

1. Go to [huggingface.co](https://huggingface.co) and sign up
2. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Click **"New token"**
4. Name it: `github-actions-deploy`
5. Type: **Write** (required for creating/updating spaces)
6. Copy the token (starts with `hf_...`)

### Step 2: Add GitHub Secrets

1. Go to your GitHub repository: `https://github.com/prashant-2050/Ai-practise`
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"** and add these three secrets:

#### Secret 1: HF_TOKEN
```
Name: HF_TOKEN
Value: hf_xxxxxxxxxxxxxxxxxxxx (your token from Step 1)
```

#### Secret 2: HF_USERNAME
```
Name: HF_USERNAME
Value: your-huggingface-username
```

#### Secret 3: SPACE_NAME
```
Name: SPACE_NAME
Value: llm-demo (or any name you prefer)
```

### Step 3: Push to Main Branch

The GitHub Action is already set up! Just push your code to the main branch:

```bash
# Make sure you're on the main branch
git checkout main

# Merge your current branch
git merge Ai-agent-tools

# Push to trigger deployment
git push origin main
```

## 🎯 What Happens Next

1. **GitHub Action triggers** automatically
2. **Files upload** to Hugging Face Spaces
3. **Space builds** and becomes live
4. **Demo is ready** at: `https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`

## 📁 Files Deployed

The action deploys these files:
- ✅ `app.py` - Entry point for HF Spaces
- ✅ `requirements.txt` - Python dependencies
- ✅ `gradio_app.py` - Your Gradio interface
- ✅ `model_config.py` - Model configuration
- ✅ `light_llm.py` - Transformer implementation
- ✅ `generate.py` - Text generation logic
- ✅ `dataset.py` - Data handling
- ✅ `README.md` - Documentation

## 🔄 Automatic Updates

Every time you push to `main` branch:
- ✅ Action runs automatically
- ✅ New code deploys
- ✅ Demo updates live
- ✅ No manual work needed!

## 🐛 Troubleshooting

### Action Failed?
1. Check **Actions** tab in your GitHub repo
2. Look at the logs for error details
3. Common issues:
   - Wrong HF_TOKEN (needs Write permission)
   - Wrong HF_USERNAME (case sensitive)
   - Missing files in repository

### Space Not Loading?
1. Go to your HF Space: `https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`
2. Check the **Logs** tab for errors
3. Common issues:
   - Missing dependencies in `requirements.txt`
   - Import errors in `app.py`
   - Model files not found

### Manual Trigger
You can manually trigger deployment:
1. Go to **Actions** tab in GitHub
2. Click **"Deploy to Hugging Face Spaces"**
3. Click **"Run workflow"**
4. Choose `main` branch and click **"Run workflow"**

## 🔗 Useful Links

- **Your Repository**: https://github.com/prashant-2050/Ai-practise
- **Your HF Space**: https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
- **HF Tokens**: https://huggingface.co/settings/tokens
- **GitHub Actions**: https://github.com/prashant-2050/Ai-practise/actions

## 🎉 Example Workflow

```bash
# 1. Make changes to your code
echo "# Updated!" >> README.md

# 2. Commit changes
git add .
git commit -m "Update demo"

# 3. Push to main (triggers deployment)
git push origin main

# 4. Check GitHub Actions tab for progress
# 5. Visit your HF Space to see updates!
```

## 💡 Pro Tips

1. **Test Locally First**: Run `python app.py` to test before pushing
2. **Check Logs**: Always check Action logs if deployment fails
3. **Version Tags**: Use tags like `v1.0` for stable releases
4. **Branch Protection**: Consider protecting main branch for production
5. **Environment Specific**: You can create separate spaces for dev/prod

## 🆘 Need Help?

If you run into issues:
1. Check the **troubleshooting** section above
2. Look at **GitHub Actions logs**
3. Check **Hugging Face Space logs**
4. Verify all **secrets** are correctly set

Your LLM is now ready for automatic deployment! 🚀✨