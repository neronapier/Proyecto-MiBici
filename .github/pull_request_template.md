## Qué cambia

<!-- Descripción breve del cambio y de la pregunta (P1, P2, P3) o componente que toca. -->

## Cómo se probó

- [ ] `make lint`
- [ ] `make test`
- [ ] `make data` y `make verify` (si cambia la ingesta o la configuración de datos)
- [ ] `make train` (si cambia el experimento; indicar los runs en MLflow)

## Lista de verificación

- [ ] Rama `<tipo>/<descripcion>` y título en Conventional Commits
- [ ] Sin datos de MiBici, modelos, carpetas de MLflow ni credenciales
- [ ] Rutas, fechas y semillas solo en `mibici/config.py`
- [ ] README actualizado si cambian comandos o datos
