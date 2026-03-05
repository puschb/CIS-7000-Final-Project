variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-east1"
}

variable "zone" {
  description = "GCP zone (must have A100 availability)"
  type        = string
  default     = "us-east1-b"
}

variable "machine_type" {
  description = <<-EOT
    Compute Engine machine type. Common options:
      a2-ultragpu-1g  = 1x A100 80GB (fine-tuning)
      a2-ultragpu-2g  = 2x A100 80GB (multi-GPU fine-tuning)
      a2-ultragpu-4g  = 4x A100 80GB (multi-GPU fine-tuning)
      a2-highgpu-1g   = 1x A100 40GB (inference)
      n1-standard-8   = no built-in GPU, use with guest_accelerator for T4/V100
  EOT
  type        = string
  default     = "a2-ultragpu-1g"
}

variable "boot_disk_size_gb" {
  description = "Boot disk size in GB"
  type        = number
  default     = 256
}

variable "boot_disk_type" {
  description = "Boot disk type (pd-ssd or pd-standard)"
  type        = string
  default     = "pd-ssd"
}

variable "preemptible" {
  description = "Use spot/preemptible VM (much cheaper, but can be reclaimed)"
  type        = bool
  default     = true
}

variable "gcs_bucket_name" {
  description = "GCS bucket name for data and checkpoints (must be globally unique)"
  type        = string
}

variable "gcs_location" {
  description = "GCS bucket location"
  type        = string
  default     = "US"
}

variable "vm_name" {
  description = "Name for the Compute Engine VM"
  type        = string
  default     = "aurora-vm"
}

variable "create_vm" {
  description = "Whether to create the GPU VM (set false to only create GCS bucket and supporting resources)"
  type        = bool
  default     = true
}
