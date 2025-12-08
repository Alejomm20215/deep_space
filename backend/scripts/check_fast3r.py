"""
Import sanity check for Fast3R inside the backend container.
Run with:
  docker-compose exec backend python3 backend/scripts/check_fast3r.py
"""
import sys


def check_import(module_name: str) -> bool:
    """Try to import a module and return success status."""
    try:
        __import__(module_name)
        return True
    except ImportError as e:
        print(f"  ❌ {module_name}: {e}")
        return False


def main():
    print("Python:", sys.version.split('\n')[0])
    print("\nsys.path:")
    for p in sys.path:
        print(f"  {p}")
    print()

    # List of imports required for Fast3R to work
    required_imports = [
        "fast3r",
        "fast3r.models",
        "fast3r.croco",
        "fast3r.dust3r",
        "fast3r.models.fast3r",
        "fast3r.dust3r.utils.image",
        "fast3r.dust3r.inference_multiview",
        "fast3r.models.multiview_dust3r_module",
    ]

    print("Checking Fast3R imports:")
    all_ok = True
    for module in required_imports:
        if check_import(module):
            print(f"  ✅ {module}")
        else:
            all_ok = False

    print()
    if all_ok:
        print("✅ Fast3R imports OK - ready for pose estimation!")
    else:
        print("❌ Fast3R imports FAILED")
        print("\nThis usually means the setup.py fix didn't work.")
        print("Check if /opt/fast3r exists and was installed correctly.")
        sys.exit(1)


if __name__ == "__main__":
    main()

