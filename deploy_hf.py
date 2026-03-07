"""
Deploy STYRS Solar Cell Inspector to Hugging Face Spaces
Run: python deploy_hf.py
"""

import os
import sys

try:
    from huggingface_hub import HfApi, create_repo, upload_folder, login
except ImportError:
    print("Installing huggingface_hub...")
    os.system(f"{sys.executable} -m pip install huggingface_hub -q")
    from huggingface_hub import HfApi, create_repo, upload_folder, login

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
SPACE_NAME = "styrs-solar-inspector"
DEPLOY_DIR = os.path.join(os.path.dirname(__file__), "hf_deploy")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "best_model.keras")

SDK_OPTIONS = ["docker", "gradio", "static"]

def main():
    print("=" * 60)
    print("  🚀 STYRS Deployment to Hugging Face Spaces")
    print("=" * 60)
    
    # Step 1: Login
    print("\n📋 Step 1: Authenticate with Hugging Face")
    print("   Go to: https://huggingface.co/settings/tokens")
    print("   Create a token with 'write' access")
    
    token = input("\n   Paste your HF token here: ").strip()
    if not token:
        print("   ❌ No token provided. Exiting.")
        return
    
    login(token=token)
    api = HfApi()
    user = api.whoami()
    username = user["name"]
    print(f"   ✅ Logged in as: {username}")
    
    # Step 2: Create Space
    repo_id = f"{username}/{SPACE_NAME}"
    print(f"\n📦 Step 2: Creating Space '{repo_id}'...")
    
    space_created = False
    for sdk in SDK_OPTIONS:
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="space",
                space_sdk=sdk,
                space_hardware="cpu-basic",
                exist_ok=True
            )
            print(f"   ✅ Space created with SDK '{sdk}': https://huggingface.co/spaces/{repo_id}")
            space_created = True
            break
        except Exception as e:
            print(f"   ⚠️ SDK '{sdk}' failed: {str(e)[:100]}")
            continue
    
    if not space_created:
        print("   ❌ Could not create Space with any SDK. Trying without SDK...")
        try:
            create_repo(repo_id=repo_id, repo_type="space", exist_ok=True)
            print(f"   ✅ Space created (basic): https://huggingface.co/spaces/{repo_id}")
            space_created = True
        except Exception as e:
            print(f"   ❌ Space creation failed: {e}")
            return
    
    # Step 3: Copy model to deploy dir
    print(f"\n📁 Step 3: Preparing deployment files...")
    deploy_model = os.path.join(DEPLOY_DIR, "best_model.keras")
    if not os.path.exists(deploy_model):
        if os.path.exists(MODEL_FILE):
            print(f"   Copying model ({os.path.getsize(MODEL_FILE)/1024/1024:.1f} MB)...")
            import shutil
            shutil.copy2(MODEL_FILE, deploy_model)
            print("   ✅ Model copied")
        else:
            print(f"   ❌ Model not found at {MODEL_FILE}")
            return
    else:
        print("   ✅ Model already in deploy dir")
    
    # Verify files
    for f in ['app.py', 'requirements.txt', 'best_model.keras']:
        path = os.path.join(DEPLOY_DIR, f)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024 / 1024
            print(f"   ✅ {f} ({size:.1f} MB)")
        else:
            print(f"   ❌ Missing: {f}")
            return
    
    # Step 4: Upload
    print(f"\n🔼 Step 4: Uploading to Hugging Face (this may take a few minutes)...")
    
    try:
        upload_folder(
            folder_path=DEPLOY_DIR,
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy STYRS Solar Cell Inspector v2.0"
        )
        print(f"\n{'='*60}")
        print(f"  ✅ DEPLOYMENT SUCCESSFUL!")
        print(f"{'='*60}")
        print(f"\n  🌐 Your app: https://huggingface.co/spaces/{repo_id}")
        print(f"  📱 API URL for Android: https://{username}-{SPACE_NAME}.hf.space")
        print(f"\n  ⏱️  Note: First build takes ~5-10 minutes on HF Spaces")
        print(f"  📋 Monitor build: https://huggingface.co/spaces/{repo_id}/logs")
    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        print("   Try running this script again with a fresh token.")

if __name__ == "__main__":
    main()
