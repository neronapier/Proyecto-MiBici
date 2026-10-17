# Pipeline en el VPS (entorno TC5061)

El workflow [`vps.yml`](../../.github/workflows/vps.yml) ejecuta el pipeline completo en el entorno MLOps del
curso alojado en el VPS: contenedor `runner` (imagen `tc5061-mlops:1.0`, 4 GB, 2 CPU) en la red `tc5061-net`,
junto al servidor de MLflow 3.16.1 (SQLite) con artefactos en MinIO. La UI pública, de solo lectura, está en
https://161.132.54.170.

## Flujo

1. GitHub Actions se conecta por SSH como `mlflow-ci` y envía `pipeline <sha> [completo|datos]`.
2. `sshd` fuerza `mlflow-ci-entrar`, que llama por `sudo` a `mlflow-ci-pipeline` (único comando permitido).
3. El script valida la orden, toma un candado y, dentro del contenedor `runner` (uid 1000), actualiza
   `/work/Proyecto-MiBici` al commit pedido. Rechaza commits que no estén en `main`.
4. Corre [`pipeline.sh`](pipeline.sh): lint, pruebas, ingesta, verificación de hashes, comprobación de que
   `data/manifest.json` no cambió y `train todos` contra `http://mlflow:5000`. Los runs llevan `entorno=vps-tc5061`.
5. El resumen con enlaces a cada run aparece en la página de la corrida.

Los CSV y Parquet quedan en `/work/Proyecto-MiBici/data/` del VPS (fuera de Git); las corridas siguientes
no vuelven a descargarlos. El entrenamiento usa unos 3 GB de RAM y tarda unos 8 minutos.

## Configuración en GitHub

| Tipo | Nombre | Valor |
|------|--------|-------|
| Environment | `vps-tc5061` | Solo despliega desde `main` |
| Secret | `VPS_SSH_KEY` | Clave privada ed25519 de `mlflow-ci` |
| Secret | `VPS_KNOWN_HOSTS` | `ssh-keyscan -t rsa 161.132.54.138` verificado |
| Variable | `VPS_HOST` | `161.132.54.138` |

## Instalación en el servidor (root)

Archivos en [`servidor/`](servidor/):

```bash
groupadd -f mlflow-ci
useradd -M -d / -s /bin/sh -g mlflow-ci -c "TC5061 GitHub Actions MiBici" mlflow-ci && passwd -l mlflow-ci
install -m 755 mlflow-ci-entrar /usr/local/bin/
install -m 750 mlflow-ci-pipeline /usr/local/sbin/
install -m 440 sudoers-mlflow-ci /etc/sudoers.d/mlflow-ci && visudo -c
install -m 644 61-mlflow-ci.conf /etc/ssh/sshd_config.d/ && sshd -t && systemctl reload ssh
install -m 644 mlflow-ci.pub /etc/ssh/authorized_keys/mlflow-ci
```

La cuenta no tiene shell, TTY ni túneles. Revocar: vaciar `/etc/ssh/authorized_keys/mlflow-ci`.
Registro: `journalctl -t tc5061-mibici-ci`.

A mano, desde el VPS: `cd /opt/tc5061 && docker compose --profile run run --rm -T -w /work/Proyecto-MiBici runner bash deploy/vps/pipeline.sh`.
