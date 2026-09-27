import { ConfigProvider, Layout, Menu, Typography } from 'antd'
import {
  DashboardOutlined,
  FundOutlined,
  DatabaseOutlined,
} from '@ant-design/icons'
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  useNavigate,
  useLocation,
} from 'react-router-dom'
import { ThunderboltFilled } from '@ant-design/icons'
import OperationsPage from './pages/OperationsPage'
import CurvesPage from './pages/CurvesPage'
import DispatchDetailPage from './pages/DispatchDetailPage'
import SizingPage from './pages/SizingPage'
import SizingDetailPage from './pages/SizingDetailPage'
import DataPage from './pages/DataPage'

const { Sider, Content } = Layout

const menuItems = [
  { key: '/', icon: <DashboardOutlined />, label: '运营' },
  { key: '/sizing', icon: <FundOutlined />, label: '投资' },
  { key: '/data', icon: <DatabaseOutlined />, label: '数据管理' },
]

const themeTokens = {
  token: {
    colorPrimary: '#e8722a',
    colorInfo: '#e8722a',
    colorLink: '#d2621d',
    colorText: '#2b302b',
    colorTextSecondary: '#6f746f',
    colorBgLayout: '#f8f9f7',
    colorBorder: '#e4e7e2',
    colorBorderSecondary: '#f1f3f0',
    borderRadius: 8,
    fontSize: 13,
  },
  components: {
    Layout: { siderBg: '#ffffff', bodyBg: '#f8f9f7', headerBg: '#ffffff' },
    Menu: {
      itemBg: 'transparent',
      itemSelectedBg: 'rgba(232,114,42,0.12)',
      itemSelectedColor: '#d2621d',
      itemColor: '#565c56',
      itemHoverColor: '#e8722a',
      itemHoverBg: 'rgba(232,114,42,0.08)',
    },
    Table: {
      headerBg: '#f8f9f7',
      headerColor: '#6f746f',
      rowHoverBg: 'rgba(232,114,42,0.08)',
      borderColor: '#e4e7e2',
      headerSplitColor: 'transparent',
    },
    Card: { paddingLG: 20 },
    Tabs: { inkBarColor: '#e8722a', itemSelectedColor: '#d2621d' },
    Segmented: {
      itemSelectedBg: '#ffffff',
      itemSelectedColor: '#d2621d',
      trackBg: '#f1f3f0',
    },
    Statistic: { titleFontSize: 12 },
  },
}

function AppLayout() {
  const navigate = useNavigate()
  const location = useLocation()
  const selectedKey =
    menuItems.find(
      (item) =>
        item.key !== '/' && location.pathname.startsWith(item.key),
    )?.key ?? '/'

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        className="ps-sider"
        theme="light"
        width={216}
        style={{
          position: 'fixed',
          left: 0,
          top: 0,
          bottom: 0,
          height: '100vh',
          zIndex: 100,
          overflow: 'auto',
        }}
      >
        <div
          style={{
            position: 'sticky',
            top: 0,
            zIndex: 1,
            padding: '18px 18px 14px',
            background: '#ffffff',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
          }}
        >
          <div
            style={{
              width: 34,
              height: 34,
              borderRadius: 9,
              background: 'linear-gradient(135deg, #f0923f, #e8722a)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 2px 6px rgba(232,114,42,0.35)',
              flexShrink: 0,
            }}
          >
            <ThunderboltFilled style={{ color: '#fff', fontSize: 18 }} />
          </div>
          <div>
            <Typography.Title level={5} style={{ margin: 0, color: '#232823', fontWeight: 700 }}>
              光储一体
            </Typography.Title>
            <Typography.Text style={{ color: '#9ba19b', fontSize: 11 }}>
              储能配置优化系统
            </Typography.Text>
          </div>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[selectedKey]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>
      <Layout style={{ marginLeft: 216 }}>
        <Content style={{ padding: '24px 28px' }}>
          <div style={{ maxWidth: 1320, margin: '0 auto' }}>
            <Routes>
              <Route path="/" element={<OperationsPage />} />
              <Route path="/curves" element={<CurvesPage />} />
              <Route path="/dispatch" element={<Navigate to="/" replace />} />
              <Route path="/dispatch/:runId" element={<DispatchDetailPage />} />
              <Route path="/sizing" element={<SizingPage />} />
              <Route path="/sizing/:runId" element={<SizingDetailPage />} />
              <Route path="/data" element={<DataPage />} />
            </Routes>
          </div>
        </Content>
      </Layout>
    </Layout>
  )
}

export default function App() {
  return (
    <ConfigProvider theme={themeTokens}>
      <BrowserRouter>
        <AppLayout />
      </BrowserRouter>
    </ConfigProvider>
  )
}
