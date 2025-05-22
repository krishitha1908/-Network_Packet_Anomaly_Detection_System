import pandas as pd
import streamlit as st
from scapy.all import sniff, IP, ARP, TCP, UDP, DNS, conf
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging
import time
import smtplib
from email.mime.text import MIMEText
import threading
import numpy as np
from datetime import datetime
import queue
import altair as alt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for thread-safe operations
packet_queue = queue.Queue()
capture_running = False
sniff_thread = None
packet_lock = threading.Lock()
stop_event = threading.Event()

# Initialize session state
def init_session_state():
    if 'packets_data' not in st.session_state:
        st.session_state.packets_data = []
    if 'capture_times' not in st.session_state:
        st.session_state.capture_times = {}
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'anomaly_scores' not in st.session_state:
        st.session_state.anomaly_scores = {}
    if 'email_sent' not in st.session_state:
        st.session_state.email_sent = False
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()

init_session_state()

# Page configuration
st.set_page_config(
    layout="wide",
    page_title="Live Network Anomaly Detection",
    page_icon="📡"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .anomaly-card {
        background-color: #fff5f5;
        border-left: 5px solid #ff4b4b;
        padding: 10px;
        margin-bottom: 10px;
    }
    .normal-card {
        background-color: #f5fff7;
        border-left: 5px solid #4CAF50;
        padding: 10px;
        margin-bottom: 10px;
    }
    .protocol-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
    }
    .tcp-badge { background-color: #4e79a7; color: white; }
    .udp-badge { background-color: #f28e2b; color: white; }
    .icmp-badge { background-color: #e15759; color: white; }
    .other-badge { background-color: #76b7b2; color: white; }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📡 Network Packet Anomaly Detection")
st.markdown("""
Real-time network traffic analysis with machine learning anomaly detection.
""")

# Protocol Mapping
protocol_mapping = {
    1: "ICMP",
    6: "TCP",
    17: "UDP",
    58: "ICMPv6",
    2054: "ARP",
    53: "DNS",
    80: "HTTP",
}

# ===== SIDEBAR CONFIGURATION =====
st.sidebar.header("Configuration")

# Capture settings
capture_mode = st.sidebar.radio(
    "Capture Mode",
    ["Fixed Count", "Continuous"],
    index=0,
    help="Fixed Count: Capture specific number of packets\nContinuous: Real-time monitoring"
)

num_packets = st.sidebar.number_input(
    "Number of Packets to Capture",
    min_value=1,
    value=50,
    step=1,
    disabled=(capture_mode == "Continuous")
)

refresh_interval = st.sidebar.slider(
    "Refresh Interval (seconds)",
    1, 10, 2,
    disabled=(capture_mode == "Fixed Count")
)

# Model settings
model_choice = st.sidebar.selectbox(
    "Anomaly Detection Model",
    ["Isolation Forest", "Local Outlier Factor"],
    index=0
)

contamination = st.sidebar.slider(
    "Outlier Fraction",
    min_value=0.01,
    max_value=0.5,
    value=0.05,
    step=0.01,
    help="Expected proportion of outliers in the data"
)

# Filter settings
st.sidebar.header("Filters")
filter_protocol = st.sidebar.selectbox(
    "Protocol Filter",
    ["All", "TCP (6)", "UDP (17)", "ICMP (1)", "ICMPv6 (58)", "ARP (2054)", "DNS (53)", "HTTP (80)"],
    index=0
)

filter_size = st.sidebar.slider(
    "Packet Size Range (bytes)",
    min_value=0,
    max_value=1500,
    value=(0, 1500)
)

# Alert settings
st.sidebar.header("Alerting")
enable_alerts = st.sidebar.checkbox("Enable Alerts", value=True)

if enable_alerts:
    email_alerts = st.sidebar.checkbox("Enable Email Alerts", value=False)
    if email_alerts:
        receiver_email = st.sidebar.text_input(
            "Notification Email",
            placeholder="your@email.com",
            help="Email address to receive anomaly alerts"
        )

        # Test email button
        if st.sidebar.button("Test Email"):
            try:
                send_email(
                    "Network Monitor Test",
                    "This is a test email from the Network Anomaly Detection System.",
                    "mladybuglore@gmail.com",
                    "yxvjcgzrjdqhfing",
                    receiver_email
                )
                st.sidebar.success("Test email sent successfully!")
            except Exception as e:
                st.sidebar.error(f"Failed to send test email: {str(e)}")

# Interface selection
interface = st.sidebar.text_input(
    "Network Interface",
    placeholder="eth0, en0, etc. (leave blank for default)",
    help="Specify the network interface to capture from"
)

# ===== PACKET PROCESSING FUNCTIONS =====
def packet_handler(packet):
    """Process each captured packet and add to queue"""
    try:
        packet_data = {}

        if packet.haslayer(IP):
            packet_data["Time"] = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            packet_data["Source_IP"] = packet[IP].src
            packet_data["Destination_IP"] = packet[IP].dst
            packet_data["Packet_Size"] = len(packet)
            packet_data["Protocol_Number"] = packet[IP].proto
            packet_data["Protocol_Name"] = protocol_mapping.get(packet[IP].proto, f"Unknown ({packet[IP].proto})")

            if packet.haslayer(TCP):
                packet_data["Source_Port"] = packet[TCP].sport
                packet_data["Destination_Port"] = packet[TCP].dport
            elif packet.haslayer(UDP):
                packet_data["Source_Port"] = packet[UDP].sport
                packet_data["Destination_Port"] = packet[UDP].dport
            else:
                packet_data["Source_Port"] = None
                packet_data["Destination_Port"] = None

        elif packet.haslayer(ARP):
            packet_data["Time"] = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            packet_data["Source_IP"] = packet[ARP].psrc
            packet_data["Destination_IP"] = packet[ARP].pdst
            packet_data["Packet_Size"] = len(packet)
            packet_data["Protocol_Number"] = 2054
            packet_data["Protocol_Name"] = "ARP"
            packet_data["Source_Port"] = None
            packet_data["Destination_Port"] = None

        elif packet.haslayer(DNS):
            packet_data["Time"] = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            if packet.haslayer(IP):
                packet_data["Source_IP"] = packet[IP].src
                packet_data["Destination_IP"] = packet[IP].dst
            else:
                packet_data["Source_IP"] = "Unknown"
                packet_data["Destination_IP"] = "Unknown"
            packet_data["Packet_Size"] = len(packet)
            packet_data["Protocol_Number"] = 53
            packet_data["Protocol_Name"] = "DNS"
            if packet.haslayer(UDP):
                packet_data["Source_Port"] = packet[UDP].sport
                packet_data["Destination_Port"] = packet[UDP].dport
            else:
                packet_data["Source_Port"] = None
                packet_data["Destination_Port"] = None
        else:
            return

        with packet_lock:
            packet_queue.put(packet_data)

    except Exception as e:
        logger.error(f"Error processing packet: {e}")

def process_queue():
    """Process packets from queue and update session state"""
    processed_packets = []
    while not packet_queue.empty():
        try:
            with packet_lock:
                packet_data = packet_queue.get()
            processed_packets.append(packet_data)
        except Exception as e:
            logger.error(f"Error processing queue: {e}")

    if processed_packets:
        with packet_lock:
            st.session_state.packets_data.extend(processed_packets)

def process_and_detect_anomalies(df):
    """Perform anomaly detection on captured packets"""
    if len(df) < 2:
        st.warning("Not enough packets for anomaly detection (need at least 2)")
        df["Anomaly_Score"] = 1
        return df

    # Feature engineering
    df["Source_IP_Hash"] = df["Source_IP"].apply(lambda x: hash(x) % (10 ** 6))
    df["Destination_IP_Hash"] = df["Destination_IP"].apply(lambda x: hash(x) % (10 ** 6))
    features = df[["Source_IP_Hash", "Destination_IP_Hash", "Packet_Size", "Protocol_Number"]]

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Anomaly detection
    if model_choice == "Local Outlier Factor":
        n_neighbors = min(20, len(df) - 1)
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        anomaly_scores = model.fit_predict(scaled_features)
    else:  # Isolation Forest
        model = IsolationForest(contamination=contamination, random_state=42)
        anomaly_scores = model.fit_predict(scaled_features)

    df["Anomaly_Score"] = anomaly_scores
    return df

def send_email(subject, body, sender_email, sender_password, receiver_email):
    """Send email alert"""
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        logger.info("Email alert sent successfully")
        st.session_state.email_sent = True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        st.session_state.email_sent = False
        raise

def handle_alerts(df):
    """Generate alerts for anomalies"""
    anomalies = df[df["Anomaly_Score"] == -1]
    if not anomalies.empty and enable_alerts:
        for _, anomaly in anomalies.iterrows():
            alert_message = (
                f"🚨 Anomaly Detected at {anomaly['Time']}\n"
                f"• Source: {anomaly['Source_IP']}:{anomaly.get('Source_Port', 'N/A')}\n"
                f"• Destination: {anomaly['Destination_IP']}:{anomaly.get('Destination_Port', 'N/A')}\n"
                f"• Protocol: {anomaly['Protocol_Name']}\n"
                f"• Size: {anomaly['Packet_Size']} bytes"
            )
            st.session_state.alerts.append(alert_message)

            if email_alerts and receiver_email and not st.session_state.email_sent:
                try:
                    send_email(
                        "NETWORK ALERT: Anomaly Detected",
                        alert_message,
                        "mladybuglore@gmail.com",
                        "yxvjcgzrjdqhfing",
                        receiver_email
                    )
                except Exception as e:
                    logger.error(f"Failed to send email alert: {e}")

def create_protocol_chart(df):
    """Create protocol distribution chart"""
    protocol_counts = df['Protocol_Name'].value_counts().reset_index()
    protocol_counts.columns = ['Protocol', 'Count']
    return alt.Chart(protocol_counts).mark_arc().encode(
        theta='Count',
        color='Protocol',
        tooltip=['Protocol', 'Count']
    ).properties(height=300)

def display_ui():
    """Display the main UI components"""
    process_queue()  # Process any queued packets

    if not st.session_state.packets_data:
        st.info("No packets captured yet")
        return

    df = pd.DataFrame(st.session_state.packets_data)

    # Apply filters
    if filter_protocol != "All":
        proto_num = int(filter_protocol.split("(")[-1].strip(")"))
        df = df[df["Protocol_Number"] == proto_num]

    df = df[
        (df["Packet_Size"] >= filter_size[0]) &
        (df["Packet_Size"] <= filter_size[1])
        ]

    if df.empty:
        st.warning("No packets match the current filters")
        return

    # Process anomalies
    df = process_and_detect_anomalies(df)
    handle_alerts(df)

    # Main display
    st.subheader("📊 Captured Packets")

    # Highlight anomalies
    def highlight_anomaly(row):
        return ['background-color: #ffe6e6' if row.Anomaly_Score == -1 else '' for _ in row]

    st.dataframe(
        df.style.apply(highlight_anomaly, axis=1),
        use_container_width=True,
        height=400
    )

    # Download button
    st.download_button(
        label="Download Packet Data",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='network_packets.csv',
        mime='text/csv'
    )

    # Analytics section
    st.subheader("📈 Traffic Analytics")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Packets", len(df))
    col2.metric("Avg Packet Size", f"{df['Packet_Size'].mean():.1f} bytes")
    col3.metric("Anomalies Detected", sum(df["Anomaly_Score"] == -1))

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(create_protocol_chart(df), use_container_width=True)

    with col2:
        st.subheader("Packet Size Distribution")
        size_chart = alt.Chart(df).mark_bar().encode(
            alt.X("Packet_Size:Q", bin=True, title="Packet Size (bytes)"),
            y='count()',
            color=alt.condition(
                alt.datum.Anomaly_Score == -1,
                alt.value('red'),
                alt.value('steelblue')
            ),
            tooltip=['count()']
        ).properties(height=300)
        st.altair_chart(size_chart, use_container_width=True)

    # Anomaly visualization
    st.subheader("Anomaly Detection")
    df["Index"] = np.arange(len(df))
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x='Index',
        y='Packet_Size',
        color=alt.condition(
            alt.datum.Anomaly_Score == -1,
            alt.value('red'),
            alt.value('green')
        ),
        tooltip=['Time', 'Source_IP', 'Destination_IP', 'Protocol_Name', 'Packet_Size']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    # Alerts section
    if st.session_state.alerts:
        st.subheader("🚨 Alerts")
        for alert in reversed(st.session_state.alerts[-5:]):  # Show last 5 alerts
            st.warning(alert)

def capture_packets_fixed_count(count):
    """Capture a fixed number of packets"""
    try:
        st.info(f"Capturing {count} packets...")
        start_time = time.time()

        # Clear previous data
        with packet_lock:
            st.session_state.packets_data = []
            st.session_state.alerts = []
            st.session_state.email_sent = False
            while not packet_queue.empty():
                packet_queue.get()  # Clear the queue

        sniff(
            prn=packet_handler,
            count=count,
            store=False,
            iface=interface if interface else None
        )

        end_time = time.time()
        st.session_state.capture_times[count] = round(end_time - start_time, 2)
        st.success(f"✅ Packet capture completed in {st.session_state.capture_times[count]} seconds!")

        # Process the captured packets
        process_queue()

    except Exception as e:
        st.error(f"Capture failed: {str(e)}")


def continuous_capture_ui(refresh_rate):
    """Continuous capture mode with live updating UI"""
    placeholder = st.empty()

    # Clear previous data
    with packet_lock:
        st.session_state.packets_data = []
        st.session_state.alerts = []
        st.session_state.email_sent = False
        while not packet_queue.empty():
            packet_queue.get()  # Clear the queue

    global capture_running
    capture_running = True

    def sniff_worker():
        """Background thread for packet capture"""
        try:
            sniff(
                prn=packet_handler,
                store=False,
                stop_filter=lambda x: not capture_running,
                iface=interface if interface else None
            )
        except Exception as e:
            logger.error(f"Sniffing error: {e}")
            global capture_running
            capture_running = False

    # Start capture thread
    sniff_thread = threading.Thread(target=sniff_worker, daemon=True)
    sniff_thread.start()

    try:
        while capture_running:
            with placeholder.container():
                # Process any queued packets
                process_queue()

                if not st.session_state.packets_data:
                    st.info("Capturing packets...")
                    time.sleep(refresh_rate)
                    continue

                df = pd.DataFrame(st.session_state.packets_data)

                # Apply filters
                if filter_protocol != "All":
                    proto_num = int(filter_protocol.split("(")[-1].strip(")"))
                    df = df[df["Protocol_Number"] == proto_num]

                df = df[
                    (df["Packet_Size"] >= filter_size[0]) &
                    (df["Packet_Size"] <= filter_size[1])
                    ]

                if df.empty:
                    st.warning("No packets match the current filters")
                    time.sleep(refresh_rate)
                    continue

                # Process anomalies
                df = process_and_detect_anomalies(df)
                handle_alerts(df)

                # Display metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Packets", len(df))
                col2.metric("Avg Packet Size", f"{df['Packet_Size'].mean():.1f} bytes")
                col3.metric("Anomalies Detected", sum(df["Anomaly_Score"] == -1))

                # Packet display
                st.subheader("📦 Latest Packets")
                st.dataframe(
                    df[['Time', 'Source_IP', 'Destination_IP', 'Protocol_Name', 'Packet_Size', 'Anomaly_Score']],
                    use_container_width=True,
                    height=300
                )

                # Anomaly visualization
                st.subheader("📈 Anomaly Detection")
                df["Index"] = np.arange(len(df))
                chart = alt.Chart(df).mark_circle(size=60).encode(
                    x='Index',
                    y='Packet_Size',
                    color=alt.condition(
                        alt.datum.Anomaly_Score == -1,
                        alt.value('red'),
                        alt.value('green')
                    ),
                    tooltip=['Time', 'Source_IP', 'Destination_IP', 'Protocol_Name', 'Packet_Size']
                ).interactive()
                st.altair_chart(chart, use_container_width=True)

                # Alerts display
                if st.session_state.alerts:
                    st.subheader("🚨 Alerts")
                    for alert in reversed(st.session_state.alerts[-5:]):  # Show last 5 alerts
                        st.error(alert)

            time.sleep(refresh_rate)

    except KeyboardInterrupt:
        pass
    finally:
        capture_running = False
        sniff_thread.join(timeout=1)
        st.success("Capture stopped")


# In your main UI control section, replace the continuous capture call with:
if capture_mode == "Fixed Count":
    if st.button("▶ Start Capture", key="start_capture"):
        capture_packets_fixed_count(num_packets)
        display_ui()
else:  # Continuous mode
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start Capture", key="start_continuous_capture"):
            continuous_capture_ui(refresh_interval)
    with col2:
        if st.button("⏹ Stop Capture", key="stop_continuous_capture"):
            #global capture_running
            capture_running = False
