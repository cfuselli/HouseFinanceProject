# HouseFinanceProject

## Overview
This project is a **Rent vs Buy calculator** written in Python. It provides a comparison between renting and buying a house using:
- Mortgage calculations (with down payment, interest rate, term).
- Costs of ownership (VvE/HOA fees, maintenance, property taxes, transaction costs).
- House appreciation and resale value after a given horizon.
- Renting costs, rent growth, and investment returns on retained capital.

The engine is structured into Python classes (`BuyAssumptions`, `RentAssumptions`, `GlobalAssumptions`, `Simulator`) and can be run either in the terminal or via a web dashboard.

A **Streamlit webapp** (`app.py`) provides sliders and plots to interactively explore scenarios.

## Deployment Setup (for future reference)

Steps taken to host on Raspberry Pi with public access:

1. **DNS**
   - Configured subdomain `rentorbuy.fuselli.net` in **Cloudflare DNS** pointing to the Raspberry Pi’s IP.
   - Added **Cloudflare DDNS** updater on the Pi so the IP stays current.

2. **Nginx Reverse Proxy**
   - Created an Nginx site config `/etc/nginx/sites-available/rentorbuy.fuselli.net` to proxy `https://rentorbuy.fuselli.net` → `localhost:8501`.
   - Used **Certbot** with Nginx plugin to get Let’s Encrypt SSL certificates.

3. **Python Virtual Environment**
   - Created a venv inside the project folder:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     pip install --upgrade pip
     pip install streamlit matplotlib numpy pandas
     ```

4. **Streamlit Config**
   - Added `~/.streamlit/config.toml`:
     ```toml
     [server]
     headless = true
     address = "127.0.0.1"
     port = 8501
     enableCORS = false

     [browser]
     serverAddress = "rentorbuy.fuselli.net"
     gatherUsageStats = false
     ```

5. **Systemd Service**
   - Created `/etc/systemd/system/rentorbuy.service`:
     ```ini
     [Unit]
     Description=Rent-or-Buy Streamlit App
     After=network-online.target
     Wants=network-online.target

     [Service]
     User=carlo
     WorkingDirectory=/home/carlo/projects/HouseFinanceProject
     ExecStart=/home/carlo/projects/HouseFinanceProject/venv/bin/streamlit run app.py
     Restart=on-failure
     RestartSec=5
     Environment=PYTHONUNBUFFERED=1

     [Install]
     WantedBy=multi-user.target
     ```
   - Enabled and started it:
     ```bash
     sudo systemctl daemon-reload
     sudo systemctl enable rentorbuy
     sudo systemctl start rentorbuy
     ```

## Usage
- Run locally in CLI:
  ```bash
  python main.py
  ```
- Run locally with dashboard:
  ```bash
  streamlit run app.py
  ```
- Access deployed app: [https://rentorbuy.fuselli.net](https://rentorbuy.fuselli.net)
