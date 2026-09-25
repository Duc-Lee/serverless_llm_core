#include <bpf/bpf_helpers.h>
#include <linux/bpf.h>
#include <linux/tcp.h>

struct early_signal_t {
  __u32 model_id;
  __u32 payload_len;
};

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 256 * 1024);
} ringbuf SEC(".maps");

SEC("sk_msg")
int intercept_inference_request(struct sk_msg_md *msg) {
  if (msg->len < 128)
    return SK_PASS;

  char buffer[64];
  __builtin_memset(buffer, 0, sizeof(buffer));
  // đọc header để tìm model id sơ bộ (giả lập cú pháp "model_id:X")
  bpf_msg_pull_data(msg, 0, sizeof(buffer), 0);

  struct early_signal_t *signal =
      bpf_ringbuf_reserve(&ringbuf, sizeof(struct early_signal_t), 0);
  if (!signal)
    return SK_PASS;
  // gia su parse dc model ID = 1
  signal->model_id = 1;
  signal->payload_len = msg->len;
  bpf_ringbuf_submit(signal, 0);
  return SK_PASS;
}

char _license[] SEC("license") = "GPL";
