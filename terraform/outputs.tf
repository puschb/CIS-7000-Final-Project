output "vm_external_ip" {
  description = "External IP of the GPU VM"
  value       = var.create_vm ? google_compute_instance.aurora_vm[0].network_interface[0].access_config[0].nat_ip : "VM not created"
}

output "vm_name" {
  description = "Name of the GPU VM"
  value       = var.vm_name
}

output "gcs_bucket_url" {
  description = "GCS bucket URL"
  value       = "gs://${google_storage_bucket.data.name}"
}

output "ssh_command" {
  description = "SSH into the VM via gcloud"
  value       = "gcloud compute ssh ${var.vm_name} --zone ${var.zone}"
}

output "vscode_ssh_config" {
  description = "Add this to ~/.ssh/config for VS Code Remote-SSH"
  value = (
    var.create_vm
    ? join("\n", [
      "# Add to ~/.ssh/config:",
      "Host aurora-vm",
      "    HostName ${google_compute_instance.aurora_vm[0].network_interface[0].access_config[0].nat_ip}",
      "    User ${split("@", google_service_account.aurora_vm.email)[0]}",
      "    IdentityFile ~/.ssh/google_compute_engine",
    ])
    : "VM not created"
  )
}

output "sync_commands" {
  description = "Useful gsutil commands for data sync"
  value = join("\n", [
    "# Upload data to GCS:",
    "gsutil -m rsync -r ./data gs://${google_storage_bucket.data.name}/data",
    "",
    "# Download data from GCS to VM:",
    "gsutil -m rsync -r gs://${google_storage_bucket.data.name}/data ./data",
    "",
    "# Upload checkpoints to GCS:",
    "gsutil -m rsync -r ./checkpoints gs://${google_storage_bucket.data.name}/checkpoints",
  ])
}
