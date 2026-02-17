"""Simple test to verify Qt/napari setup"""
import numpy as np
print("Testing imports...")
print("✓ NumPy imported")

try:
    import napari
    print("✓ napari imported")
except Exception as e:
    print(f"✗ napari import failed: {e}")
    exit(1)

try:
    from qtpy import QtCore
    print(f"✓ Qt backend: {QtCore}")
except Exception as e:
    print(f"✗ Qt import failed: {e}")
    exit(1)

# Test napari without GUI (headless)
try:
    # Create a simple test volume
    test_volume = np.random.rand(50, 50, 50)
    print("✓ Test volume created")
    
    # Try to create viewer (may require display)
    print("\nAttempting to create napari viewer...")
    print("(If GUI opens, close the window to continue)")
    viewer = napari.Viewer()
    viewer.add_image(test_volume, name="Test")
    print("✓ Viewer created successfully!")
    napari.run()
    
except Exception as e:
    print(f"✗ Viewer creation failed: {e}")
    exit(1)

print("\n✓ All tests passed!")
