# Deployment

## Install Docker on `Oracle-Linux-9.6-aarch64-2025.10.23-0`

```shell
# Check your system
uname -a
> Linux arm-03 6.12.0-104.43.4.2.el9uek.aarch64 #1 SMP Wed Oct  8 12:27:24 PDT 2025 aarch64 aarch64 aarch64 GNU/Linux

# Uninstall old versions
sudo dnf remove docker docker-client docker-client-latest docker-common docker-latest docker-latest-logrotate docker-logrotate docker-engine

# Install the 'dnf-plugins-core' package to manage repositories
sudo dnf install -y dnf-plugins-core

# Add the official Docker repository
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

# Install Docker Engine, CLI, Containerd, and Docker Compose plugin
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin


# Start Docker Engine
sudo systemctl enable --now docker

# Add your current user to the docker group to run without 'sudo'
sudo usermod -aG docker $USER

# Refresh your group membership immediately without logging out
newgrp docker

# Verify installation by running a test container
docker run --rm hello-world
```

```shell
POST http://localhost:9200/_security/user/kibana_system/_password
Authorization: Basic elastic changeme
Content-Type: application/json

{
  "password": "changeme"
}
```