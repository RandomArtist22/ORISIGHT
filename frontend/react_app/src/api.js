const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

async function parseResponse(response) {
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = data?.detail || data?.message || 'Request failed'
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return data
}

export async function analyzeCase({ imageFile, symptoms, history, doctorMode }) {
  const formData = new FormData()
  formData.append('image', imageFile)
  formData.append('symptoms', symptoms)
  formData.append('history', history)
  formData.append('doctor_mode', String(Boolean(doctorMode)))

  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData
  })

  return parseResponse(response)
}

export async function triggerScrape(pubmedQuery) {
  const response = await fetch(`${API_BASE}/scrape`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pubmed_query: pubmedQuery, pubmed_max_results: 8 })
  })
  return parseResponse(response)
}

export async function triggerEmbed(reindex = false) {
  const response = await fetch(`${API_BASE}/embed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reindex })
  })
  return parseResponse(response)
}

export async function submitDoctorReview(caseId, review) {
  const response = await fetch(`${API_BASE}/report/${caseId}/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(review)
  })
  return parseResponse(response)
}
