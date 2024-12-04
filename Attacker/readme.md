# CAN Attacker

The `CAN Attacker` is a command-line utility for testing and simulating various attack scenarios on a Controller Area Network (CAN). It supports multiple attack types, each designed to disrupt or manipulate the CAN bus behavior. This tool is designed for research and security testing purposes only. Unauthorized use is strictly prohibited.

---

## Features

- **Attack Types**:
    - **DoS (Denial of Service)**: Floods the network with high-priority CAN messages.
    - **Fuzzing**: Sends random CAN frames to test robustness.
    - **Replay**: Captures and replays CAN frames.
    - **Fabrication**: Creates and sends new fabricated CAN frames.
    - **Malfunction**: Sends abnormal data using a normal CAN ID to disrupt systems.
    - **Spoofing**: Imitates an existing ECU by sending frames with valid IDs but false data.
    - **Masquerade**: Hijacks an ECU's identity by stealing its CAN ID.
- **Frequency Options**:
    - **High**: Sends messages at 10-50ms intervals.
    - **Medium**: Sends messages at 100-500ms intervals.
    - **Low**: Sends messages at 1-5s intervals.
    - **Adaptive**: Dynamically adjusts message frequency based on the target CAN ID's observed traffic.
- **Frame Recording**: Supports capturing and replaying up to 100 frames from the CAN bus for testing replay attacks.

---

## Requirements

- Linux with CAN socket support.
- C++17 compatible compiler (e.g., `g++`).
- A connected and configured CAN interface (e.g., `can0`).

---

## Compilation

Use the following command to compile the program:

```bash
g++ -o can_attacker can_attacker.cpp -std=c++17 -pthread
```

---

## Usage

### Syntax

```bash
./can_attacker <attack_type> <frequency> [can_id]
```

### Arguments

- `<attack_type>`:
    - `dos`: Denial of Service attack.
    - `fuzzing`: Fuzzing attack.
    - `replay`: Replay attack.
    - `fabrication`: Fabrication attack.
    - `malfunction`: Malfunction attack.
    - `spoofing`: Spoofing attack.
    - `masquerade`: Masquerade attack.
- `<frequency>`:
    - `high`: 10-50ms intervals.
    - `medium`: 100-500ms intervals.
    - `low`: 1-5s intervals.
    - `adaptive`: Adjusts dynamically based on target CAN ID traffic.
- `[can_id]`:
    - A hexadecimal value representing the target CAN ID (e.g., `0x123`).
    - Required for `malfunction`, `spoofing`, `masquerade`, and `adaptive` frequency attacks.

### Examples

1. **DoS attack with high frequency**:
    
    ```bash
    ./can_attacker dos high
    ```
    
2. **Fuzzing attack with adaptive frequency for a specific CAN ID**:
    
    ```bash
    ./can_attacker fuzzing adaptive 0x123
    ```
    
3. **Replay attack with medium frequency**:
    
    ```bash
    ./can_attacker replay medium
    ```
    
4. **Fabrication attack with low frequency targeting a specific CAN ID**:
    
    ```bash
    ./can_attacker fabrication low 0x456
    ```
    

---

## Features in Detail

### Frame Recording

- The `replay` attack automatically captures 100 frames from the CAN bus before replaying them.
- Frames are stored in memory and replayed sequentially.

## Tested Critical CAN IDs and Attack Scenarios

### 1. EPAS3S_sysStatus (0x370)
**Target**: Electronic Power Steering System
```bash
./can_attacker spoofing adaptive 0x370
```
**Impact**:
- Manipulation of steering angle sensor data
- False hands-on level reporting
- Potential disruption of power steering assistance

### 2. DI_torque (0x108)
**Target**: Torque Management System
```bash
./can_attacker malfunction adaptive 0x108
```
**Impact**:
- Incorrect torque readings
- False axle speed reporting
- Potential drivetrain instability

### 3. DI_systemStatus (0x118)
**Target**: Vehicle System Status
```bash
./can_attacker masquerade adaptive 0x118
```
**Impact**:
- False gear position reporting
- Manipulated traction control states
- Incorrect brake pedal status

### 4. ESP_B (0x155)
**Target**: Vehicle Stability Control
```bash
./can_attacker spoofing adaptive 0x155
```
**Impact**:
- False vehicle speed reporting
- Manipulated wheel rotation status
- Potential stability control malfunction

### 5. DI_speed (0x257)
**Target**: Speed Management System
```bash
./can_attacker malfunction adaptive 0x257
```
**Impact**:
- Incorrect speed display
- False speed unit conversion
- Potential cruise control disruption

### 6. DAS_status2 (0x389)
**Target**: Driver Assistance Systems
```bash
./can_attacker masquerade adaptive 0x389
```
**Impact**:
- False collision warnings
- Manipulated ACC status
- Disrupted driver assistance features

## Adaptive Attack Strategy
- Monitors natural message frequency of each target ID
- Dynamically adjusts attack timing to match or slightly exceed normal traffic
- Maintains persistent attack while avoiding detection
- Exploits specific vulnerabilities in each system

---

## Signals

- **CTRL+C**: Safely interrupts and stops the program.

---

## Notes

- Ensure the CAN interface (e.g., `can0`) is configured and operational before using this tool.
- Malfunction, Spoofing, and Masquerade attacks require a valid CAN ID.

---

## Disclaimer

This software is intended for research and security testing in authorized environments only. Unauthorized use on live or production systems is illegal and strictly prohibited. Use responsibly.