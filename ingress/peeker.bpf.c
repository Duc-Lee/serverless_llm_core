#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/in.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

struct inference_trigger_event {
    __u32 saddr;
    __u32 daddr;
    __u16 sport;
    __u16 dport;
    __u32 model_hash;
    __u32 estimated_len;
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} trigger_events SEC(".maps");

static __always_inline __u32 fnv1a_32(const char *data, int len) {
    __u32 hash = 2166136261U;
    for (int i = 0; i < len && i < 32; i++) {
        hash ^= (unsigned char)data[i];
        hash *= 16777619U;
    }
    return hash;
}

SEC("tc")
int hydra_early_peeker(struct __sk_buff *skb) {
    void *data_end = (void *)(long)skb->data_end;
    void *data = (void *)(long)skb->data;

    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end || eth->h_proto != bpf_htons(ETH_P_IP)) return TC_ACT_OK;

    struct iphdr *iph = (void *)(eth + 1);
    if ((void *)(iph + 1) > data_end || iph->protocol != IPPROTO_TCP) return TC_ACT_OK;

    struct tcphdr *tcph = (void *)iph + (iph->ihl * 4);
    if ((void *)(tcph + 1) > data_end || tcph->dest != bpf_htons(8080)) return TC_ACT_OK;

    __u32 tcp_header_len = tcph->doff * 4;
    unsigned char *payload = (unsigned char *)tcph + tcp_header_len;
    
    if ((void *)(payload + 16) > data_end) return TC_ACT_OK;

    if (payload[0] == 'P' && payload[1] == 'O' && payload[2] == 'S' && payload[3] == 'T') {
        struct inference_trigger_event *event;
        event = bpf_ringbuf_reserve(&trigger_events, sizeof(*event), 0);
        if (!event) return TC_ACT_OK;

        event->saddr = iph->saddr;
        event->daddr = iph->daddr;
        event->sport = bpf_ntohs(tcph->source);
        event->dport = bpf_ntohs(tcph->dest);
        event->estimated_len = bpf_ntohs(iph->tot_len) - (iph->ihl * 4) - tcp_header_len;
        event->model_hash = fnv1a_32("llama-3-8b", 10); 

        bpf_ringbuf_submit(event, 0);
    }
    return TC_ACT_OK;
}
char _license[] SEC("license") = "GPL";
