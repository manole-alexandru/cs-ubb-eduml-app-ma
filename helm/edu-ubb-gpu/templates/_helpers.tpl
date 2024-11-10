{{- define "deployment.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "deployment.namespace" -}}
{{- default "mlflow" .Values.namespaceOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "deployment.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "deployment.labels" -}}
helm.sh/chart: {{ include "deployment.chart" . }}
{{ include "deployment.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "deployment.selectorLabels" -}}
app.kubernetes.io/name: {{ include "deployment.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "image.repository" -}}
  {{- if hasKey .Values "image" -}}
    {{- if hasKey .Values.image "repository" -}}
      {{- .Values.image.repository -}}
    {{- else -}}
      {{- fail "Error: 'image.repository' is required. Please add it to your values.yaml" -}}
    {{- end -}}
  {{- else -}}
    {{- fail "Error: 'image.repository' is required. Please define 'image' object in your values.yaml" -}}
  {{- end -}}
{{- end -}}

{{- define "image.tag" -}}
  {{- if hasKey .Values "image" -}}
    {{- .Values.image.tag | default "latest" -}}
  {{- else -}}
    "main"
  {{- end -}}
{{- end -}}

{{- define "image.uri" -}}
{{- printf "%s:%s" (include "image.repository" .) (include "image.tag" .) -}}
{{- end -}}

{{- define "experiment.name" -}}
{{- if .Values.experiment }}
{{ .Values.experiment }}
{{- else }}
{{- printf "%s-experiment" (include "deployment.name" .) -}}
{{- end }}
{{- end -}}