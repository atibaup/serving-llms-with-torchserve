terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "4.51.0"
    }
  }
}

variable "gcp_project_id" {
 type = string
 description = "GCP project ID"
}

variable "gcp_region" {
 type = string
 default = "europe-west4"
 description = "GCP region"
}

provider "google-beta" {
  project     = var.gcp_project_id
  region      = var.gcp_region
}

provider "google" {
  project     = var.gcp_project_id
  region      = var.gcp_region
}

resource "google_project_service" "cloud_resource_manager" {
  service = "cloudresourcemanager.googleapis.com"
  disable_dependent_services = true
}

resource "google_project_service" "vertexai_service" {
  service = "aiplatform.googleapis.com"
  disable_dependent_services = true
  depends_on = [google_project_service.cloud_resource_manager]
}

resource "google_project_service" "container_registry" {
  service = "containerregistry.googleapis.com"
  disable_dependent_services = true
  depends_on = [google_project_service.cloud_resource_manager]
}

resource "google_service_account" "vertexai_sa" {
  account_id   = "vertexai"
  display_name = "Service account for vertex AI "
}

resource "google_project_iam_binding" "vertexai_iam" {
  project = var.gcp_project_id
  role = "roles/aiplatform.user"
  members = [
      format("serviceAccount:%s", google_service_account.vertexai_sa.email)
    ]
}

data "google_iam_policy" "storage_admin" {
  binding {
    role = "roles/storage.admin"
    members = [
      format("serviceAccount:%s", google_service_account.vertexai_sa.email)
    ]
  }
}

resource "google_storage_bucket" "vertexai_bucket" {
  name          = format("%s-vertexai-staging", var.gcp_project_id)
  location      = "EU"
  force_destroy = true
  public_access_prevention = "enforced"
  uniform_bucket_level_access = true
}

resource "google_storage_bucket_iam_policy" "vertexai_bucket_iam" {
  bucket = google_storage_bucket.vertexai_bucket.name
  policy_data = data.google_iam_policy.storage_admin.policy_data
}

resource "google_vertex_ai_metadata_store" "default-store" {
  provider      = google-beta
  name          = "default"
  description   = "Store to experiment with vertexai deployments"
  region        = "europe-west4"
  depends_on = [google_project_service.vertexai_service]
}