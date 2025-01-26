module.exports = {
    apps: [{
      name: "phi2-api",
      interpreter: "/opt/phi2-api/venv/bin/python",
      script: "script.py",
      env: {
        PATH: "/opt/phi2-api/venv/bin:$PATH"
      }
    }]
  }