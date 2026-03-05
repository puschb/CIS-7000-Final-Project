terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# --------------------------------------------------------------------------
# GCS Bucket — persists independently of the VM
# --------------------------------------------------------------------------

resource "google_storage_bucket" "data" {
  name     = var.gcs_bucket_name
  location = var.gcs_location

  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = false
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
}

# --------------------------------------------------------------------------
# Service Account — gives VM access to GCS
# --------------------------------------------------------------------------

resource "google_service_account" "aurora_vm" {
  account_id   = "aurora-vm-sa"
  display_name = "Aurora VM Service Account"
}

resource "google_project_iam_member" "storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.aurora_vm.email}"
}

# --------------------------------------------------------------------------
# Firewall — allow SSH
# --------------------------------------------------------------------------

resource "google_compute_firewall" "allow_ssh" {
  name    = "allow-ssh-aurora"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["aurora-vm"]
}

# --------------------------------------------------------------------------
# GPU VM — only created when var.create_vm is true
# --------------------------------------------------------------------------

resource "google_compute_instance" "aurora_vm" {
  count = var.create_vm ? 1 : 0

  name         = var.vm_name
  machine_type = var.machine_type
  zone         = var.zone

  tags = ["aurora-vm"]

  boot_disk {
    initialize_params {
      image = "deeplearning-platform-release/common-cu124"
      size  = var.boot_disk_size_gb
      type  = var.boot_disk_type
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  scheduling {
    preemptible                 = var.preemptible
    automatic_restart           = var.preemptible ? false : true
    on_host_maintenance         = "TERMINATE"
    provisioning_model          = var.preemptible ? "SPOT" : "STANDARD"
    instance_termination_action = var.preemptible ? "STOP" : null
  }

  service_account {
    email  = google_service_account.aurora_vm.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    install-nvidia-driver = "True"
  }

  metadata_startup_script = file("${path.module}/startup.sh")

  lifecycle {
    ignore_changes = [metadata["ssh-keys"]]
  }
}
