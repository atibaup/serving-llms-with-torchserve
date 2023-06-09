output "storage_bucket" {
  value = format("gs://%s", google_storage_bucket.vertexai_bucket.name)
}

output "service_account" {
  value = google_service_account.vertexai_sa.name
}