FROM nvcr.io/nvidia/pytorch:26.03-py3
WORKDIR /opt/project/build/
SHELL ["/bin/bash", "-c"]

# Rename the default user and group from ubuntu → sthornington
RUN groupmod -n sthornington ubuntu && \
    usermod -l sthornington -d /home/sthornington -m ubuntu

RUN apt-get update
RUN apt-get install -y curl build-essential graphviz emacs
RUN apt-get update && apt-get install -y --no-install-recommends \
  less \
  git \
  procps \
  sudo \
  fzf \
  zsh \
  man-db \
  unzip \
  gnupg2 \
  gh \
  iptables \
  ipset \
  iproute2 \
  dnsutils \
  aggregate \
  jq \
  nano \
  nodejs \
  npm \
  vim \
  nvtop \
  ninja-build \
  cmake \
  apt-transport-https \
  ca-certificates

# Install ClickHouse server and client
RUN curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' | gpg --dearmor -o /usr/share/keyrings/clickhouse-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg] https://packages.clickhouse.com/deb stable main" | tee /etc/apt/sources.list.d/clickhouse.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y clickhouse-server clickhouse-client 

RUN mkdir -p /usr/local/share/npm-global && \
  chown -R sthornington:sthornington /usr/local/share

ARG USERNAME=sthornington 

ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin

# Persist bash history.
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  && mkdir /commandhistory \
  && touch /commandhistory/.bash_history \
  && chown -R $USERNAME /commandhistory

ENV DEVCONTAINER=true

ARG GIT_DELTA_VERSION=0.18.2
RUN ARCH=$(dpkg --print-architecture) && \
  wget "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  sudo dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

WORKDIR /home/sthornington
ENV PATH="/home/sthornington/.local/bin:${PATH}"

RUN mkdir -p /home/sthornington/.claude && \
  echo '{"permissions":{"allow":[],"deny":[]},"bypassPermissions":true}' > /home/sthornington/.claude/settings.json && \
  chown -R sthornington:sthornington /home/sthornington/.claude

# User level setup

USER sthornington
ENV EDITOR=emacs 
ENV VISUAL=emacs 

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    . "$HOME/.cargo/env" && \
    rustc --version

COPY --chown=sthornington:sthornington ["requirements.txt", "/opt/project/build/"]

RUN pip install --user fastai==2.8.5 --no-deps
# todo can we upgrade this some time?
RUN pip install --user timm==0.6.13 --no-deps
RUN pip install --user -r /opt/project/build/requirements.txt

# Must use --no-build-isolation to use container's nvcc and PyTorch
# FORCE_BUILD required on ARM64: the packages incorrectly download x86_64 wheels
ENV MAX_JOBS=20
ENV CMAKE_BUILD_PARALLEL_LEVEL=20

# Verify optimized kernels work
WORKDIR /galaxy


#ENV CLAUDE_CODE_VERSION=2.0.64
# Install Claude
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# Install codex
RUN npm i -g @openai/codex

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8889", "--no-browser", "--ServerApp.token=", "--ServerApp.password="]
#CMD ["tail", "-f", "/dev/null"]
#CMD ["claude","--dangerously-skip-permissions"]
