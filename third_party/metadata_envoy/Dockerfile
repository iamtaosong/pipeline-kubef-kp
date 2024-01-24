# Copyright 2019 The Kubeflow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


FROM envoyproxy/envoy:v1.12.2 as builder

FROM ubuntu:22.04 as envoy

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y ca-certificates \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/* \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/envoy

COPY --from=builder /usr/local/bin/envoy /usr/local/bin/envoy
COPY --from=builder /etc/envoy/envoy.yaml /etc/envoy/envoy.yaml
COPY --from=builder /docker-entrypoint.sh /docker-entrypoint.sh
 

EXPOSE 10000

# COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["envoy", "-c", "/etc/envoy/envoy.yaml"]



 
From envoy
RUN apt-get update -y && \
  apt-get install --no-install-recommends -y -q gettext openssl

COPY envoy.yaml /etc/envoy.yaml

# Copy license files.
#RUN mkdir -p /third_party
COPY license.txt /third_party/license.txt

ENTRYPOINT ["/usr/local/bin/envoy", "-c"]
CMD ["/etc/envoy.yaml"]


# docker build  . -t kubef/envoyproxy/envoy:v1.12.2
# https://github.com/kubeflow/pipelines/blob/1.8.2/third_party/metadata_envoy/Dockerfile