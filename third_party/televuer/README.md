# TeleVuer

> **OrcaManipulation delivery note.** This is a vendored copy of Unitree TeleVuer
> `766de45e74373ae0ea66321d942ce538385655a5`, republished as `4.0.0+orca.1`. Install it
> only through `bash scripts/install_runtime.sh` in the repository root; do not clone
> upstream separately. Certificate discovery via `XR_TELEOP_CERT`, `~/.config/xr_teleoperate/`
> or the package directory has been removed. The listener defaults to `127.0.0.1` and plain
> HTTP/WS, reached from the headset through `adb reverse tcp:8012 tcp:8012`. Any other
> `host` requires explicit `cert_file` and `key_file` arguments, because WebXR only starts
> an immersive session in a secure context. The sections below are upstream documentation;
> where they describe certificate paths or `pip install -e`, the delivery rules above win.

The TeleVuer library is a specialized version of the [Vuer](https://github.com/vuer-ai/vuer) library, designed to enable XR device-based teleoperation of Unitree Robotics robots. This library acts as a wrapper for Vuer, providing additional adaptations specifically tailored for Unitree Robotics. By integrating XR device capabilities, such as hand tracking and controller tracking, TeleVuer facilitates seamless interaction and control of robotic systems in immersive environments.

Currently, this module serves as a core component of the [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) library, offering advanced functionality for teleoperation tasks. It supports various XR devices, including Apple Vision Pro, Meta Quest3, Pico 4 Ultra Enterprise etc., ensuring compatibility and ease of use for robotic teleoperation applications.

The image input of this library works in conjunction with the [teleimager](https://github.com/silencht/teleimager) library. We recommend using both libraries together.

## 0. 🔖 Release Note

### V4.0 🏷️ brings updates:

1. Improved Display Modes

    Removed the old “pass_through” mode. The system now supports three modes:

    - immersive: fully immersive mode; VR shows the robot's first-person view (zmq or webrtc must be enabled).

    - pass-through: VR shows the real world through the VR headset cameras; no image from zmq or webrtc is displayed (even if enabled).

    - ego: a small window in the center shows the robot's first-person view, while the surrounding area shows the real world.

2. Enhanced Immersion

    Adjusted the image plane height for immersive and ego modes to provide a more natural and comfortable VR experience

### V3.0 🏷️ brings updates:
1. Added `pass_through` interface to enable/disable the pass-through mode.
2. Support `webrtc` interface to enable/disable the webrtc streaming mode.
3. Use `render_to_xr` method (adjust from `set_display_image`) to send images to XR device.

### V2.0 🏷️ brings updates:

1. Image transport is now by reference instead of external shared memory.
2. Renamed the get-data function from `get_motion_state_data` to `get_tele_data`.
3. Fixed naming errors (`waist` → `wrist`)
4. Var name align with **vuer** conventions.
5. Streamlined the data structure: removed the nested `TeleStateData` and return everything in the unified `TeleData`.
6. Added new image-transport interfaces such as `set_display_image`.

## 1. 🗺️ Diagram

<p align="center">
  <a href="https://oss-global-cdn.unitree.com/static/5ae3c9ee9a3d40dc9fe002281e8aeac1_2975x3000.png">
    <img src="https://oss-global-cdn.unitree.com/static/5ae3c9ee9a3d40dc9fe002281e8aeac1_2975x3000.png" alt="Diagram" style="width: 50%;">
  </a>
</p>

## 2. 📦 Install

### 2.1 📥 Install televuer repository

In this delivery TeleVuer is already vendored and is installed by the repository script:

```bash
bash scripts/install_runtime.sh
```


### 2.2 🔑 Generate Certificate Files

The televuer module requires SSL certificates to allow XR devices (such as Pico / Quest / Apple Vision Pro) to connect securely via HTTPS / WebRTC.

1. For Pico / Quest XR Devices

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem
# Check the Generated Files
$ ls
build  cert.pem  key.pem  LICENSE  pyproject.toml  README.md  src  test
```

2. For Apple Vision Pro

```bash
openssl genrsa -out rootCA.key 2048
openssl req -x509 -new -nodes -key rootCA.key -sha256 -days 365 -out rootCA.pem -subj "/CN=xr-teleoperate"
openssl genrsa -out key.pem 2048
openssl req -new -key key.pem -out server.csr -subj "/CN=localhost"
vim server_ext.cnf
# Add the following content (make sure IP.2 matches your host machine’s IP address, which you can find with ifconfig or similar):
subjectAltName = @alt_names
[alt_names]
DNS.1 = localhost
IP.1 = 192.168.123.164
IP.2 = 192.168.123.2
# Then sign the certificate:
openssl x509 -req -in server.csr -CA rootCA.pem -CAkey rootCA.key -CAcreateserial -out cert.pem -days 365 -sha256 -extfile server_ext.cnf
# Check the Generated Files
$ ls
build  cert.pem  key.pem  LICENSE  pyproject.toml  README.md  rootCA.key  rootCA.pem  rootCA.srl  server.csr  server_ext.cnf  src  test
# Use AirDrop to copy rootCA.pem to your Apple Vision Pro device and install it manually as a trusted certificate.
```

3. 🧱 Allow Firewall Access

```bash
sudo ufw allow 8012
```

### 2.3 🔐 Configure Certificate Paths (delivery behaviour)

In this vendored copy the certificate paths are arguments, never discovered from the
environment or the filesystem. Pass them together:

```python
TeleVuerWrapper(..., host="192.168.123.2", cert_file="/path/cert.pem", key_file="/path/key.pem")
```

From the collection entry point the equivalent flags are `--tv_host`, `--tv_cert_file` and
`--tv_key_file`. Omit all three to get the default loopback HTTP/WS listener used with
`adb reverse`. Certificates and private keys are never stored in this repository.

## 3. 🧐 Test

```bash
python test_televuer.py 
# or 
python test_tv_wrapper.py

# First, use Apple Vision Pro or Pico 4 Ultra Enterprise to connect to the same Wi-Fi network as your computer.
# Next, open safari / pico browser, enter https://host machine's ip:8012/?ws=wss://host machine's ip:8012
# for example, https://192.168.123.2:8012?ws=wss://192.168.123.2:8012
# Use the appropriate method (hand gesture or controller) to click the "pass-through" button in the bottom-left corner of the screen.

# Press Enter in the terminal to launch the program.
```

## 4. 📌 Version History

`vuer==0.0.32rc7`

- **Functionality**:
  - Hand tracking works fine.
  - Controller tracking is not supported.

---

`vuer==0.0.35`

- **Functionality**:
  - AVP hand tracking works fine.
  - PICO hand tracking works fine, but the right eye occasionally goes black for a short time at startup.

---

`vuer==0.0.36rc1` to `vuer==0.0.42rc16`

- **Functionality**:
  - Hand tracking only shows a flat RGB image (no stereo view).
  - PICO hand and controller tracking behave the same, with occasional right-eye blackouts at startup.
  - Hand or controller markers are displayed as either black boxes (`vuer==0.0.36rc1`) or RGB three-axis color coordinates (`vuer==0.0.42rc16`).

---

`vuer==0.0.42` to `vuer==0.0.45`

- **Functionality**:
  - Hand tracking only shows a flat RGB image (no stereo view).
  - Unable to retrieve hand tracking data.
  - Controller tracking only shows a flat RGB image (no stereo view), but controller data can be retrieved.

---

`vuer==0.0.46` to `vuer==0.0.56`

- **Functionality**:
  - AVP hand tracking works fine.
  - In PICO hand tracking mode:
    - Using a hand gesture to click the "Virtual Reality" button causes the screen to stay black and stuck loading.
    - Using the controller to click the button works fine.
  - In PICO controller tracking mode:
    - Using the controller to click the "Virtual Reality" button causes the screen to stay black and stuck loading.
    - Using a hand gesture to click the button works fine.
  - Hand marker visualization is displayed as RGB three-axis color coordinates.

---

`vuer==0.0.60`
- **Recommended Version**

- **Functionality**:
  - Stable functionality with good compatibility.
  - Most known issues have been resolved.
  - A minor issue: the right eye briefly goes black right after startup.
- **References**:
  - [GitHub Issue #53](https://github.com/unitreerobotics/xr_teleoperate/issues/53)
  - [GitHub Issue #45](https://github.com/vuer-ai/vuer/issues/45)
  - [GitHub Issue #65](https://github.com/vuer-ai/vuer/issues/65)

---

## Tested Devices
please refer to our wiki doc [XR Device](https://github.com/unitreerobotics/xr_teleoperate/wiki/XR_Device)

## Notes
- **Recommended Version**: Use `vuer==0.0.60` for the best functionality and stability.
- **Black Screen Issue**: On PICO devices, choose the appropriate interaction method (hand gesture or controller) based on the mode to avoid black screen issues.
